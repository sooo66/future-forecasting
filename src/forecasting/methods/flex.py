"""Simplified FLEX-style forecasting method."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any

from forecasting.contracts import ForecastMethod, MethodArtifact, MethodRuntimeContext, MethodSession, QuestionRecord
from forecasting.embeddings import DEFAULT_EMBEDDING_MODEL, build_text_embedder
from forecasting.memory import FlexExperience, FlexLibrary
from forecasting.methods._agentic_shared import (
    build_memory_query,
    coerce_config,
    question_cutoff_time,
    run_agentic_forecast,
)
from forecasting.prompts import build_flex_distill_messages
from forecasting.question_tools import FlexMemoryTool, ResidentCodeInterpreterTool


@dataclass(frozen=True)
class FlexConfig:
    agent_max_steps: int = 5
    search_top_k: int = 3
    strategy_top_k: int = 5
    pattern_top_k: int = 5
    case_top_k: int = 5
    preload_zone: str = "golden"
    preload_domain_match: bool = True
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL
    embedding_device: str | None = None
    merge_similarity_threshold: float = 0.92
    artifact_filename: str = "flex_mem.jsonl"


class FlexMethod(ForecastMethod):
    method_id = "flex"

    def build_session(self, runtime_ctx: MethodRuntimeContext, method_config: Any | None = None) -> MethodSession:
        return _FlexSession(runtime_ctx, coerce_config(method_config, FlexConfig))


class _FlexSession(MethodSession):
    def __init__(self, runtime_ctx: MethodRuntimeContext, config: FlexConfig) -> None:
        self._runtime_ctx = runtime_ctx
        self._config = config
        self._llm = runtime_ctx.make_llm()
        self._library = FlexLibrary(
            embedder=build_text_embedder(
                model_name=config.embedding_model_name,
                device=config.embedding_device,
            ),
            model_name=config.embedding_model_name,
            merge_similarity_threshold=config.merge_similarity_threshold,
        )
        self._code_interpreter = ResidentCodeInterpreterTool(
            work_dir=runtime_ctx.project_root / ".qwen_agent_workspace" / "forecasting" / "flex"
        )

    def run_question(self, question: QuestionRecord):
        cutoff_time = question_cutoff_time(question)
        preloaded = self._library.retrieve_default_bundle(
            build_memory_query(question),
            open_time=cutoff_time,
            per_level={
                "strategy": self._config.strategy_top_k,
                "pattern": self._config.pattern_top_k,
                "case": self._config.case_top_k,
            },
            zone=(self._config.preload_zone or "").strip().lower() or None,
            domain=question["domain"] if self._config.preload_domain_match else None,
        )
        result = run_agentic_forecast(
            question,
            self._llm,
            self._runtime_ctx.search_engine,
            project_root=self._runtime_ctx.project_root,
            method_name="flex",
            agent_max_steps=self._config.agent_max_steps,
            search_top_k=self._config.search_top_k,
            flex_memory_tool=FlexMemoryTool(self._library, cutoff_time=cutoff_time),
            flex_preloaded=preloaded,
            code_interpreter_tool=self._code_interpreter,
        )
        if "error" not in result:
            self._library.queue_many(_build_flex_experiences(question, result, llm=self._llm))
        return result

    def finalize(self) -> list[MethodArtifact]:
        return [
            MethodArtifact(
                filename=self._config.artifact_filename,
                format="jsonl",
                payload=self._library.artifact_rows(),
            ),
        ]


def _build_flex_experiences(
    question: QuestionRecord,
    result: dict[str, Any],
    *,
    llm: Any | None = None,
) -> list[FlexExperience]:
    correctness = _is_correct_prediction(question, result)
    zone = "golden" if correctness else "warning"
    payload = _distill_flex(question, result, correctness=correctness, llm=llm)
    if not payload:
        payload = _default_flex_distillation(question, result, correctness=correctness)
    payload = _sanitize_flex_payload(payload, question=question, correctness=correctness, result=result)
    items: list[FlexExperience] = []
    for level in ["strategy", "pattern", "case"]:
        block = payload.get(level) or {}
        items.append(
            FlexExperience(
                experience_id=f"flex-{zone}-{level}-{question['market_id']}",
                source_question_id=question["market_id"],
                domain=question["domain"],
                zone=zone,
                level=level,
                title=str(block.get("title") or f"{question['domain']} {level}").strip(),
                summary=str(block.get("summary") or "").strip() or _default_summary(question, level, zone),
                content=str(block.get("content") or "").strip() or _default_content(result, zone),
                created_at=question["resolve_time"],
                source_open_time=question["open_time"],
                source_resolved_time=question["resolve_time"],
                outcome=int(question["label"]),
                correctness=correctness,
                support_count=1,
            )
        )
    return items


def _distill_flex(
    question: QuestionRecord,
    result: dict[str, Any],
    *,
    correctness: bool,
    llm: Any | None,
) -> dict[str, Any]:
    if llm is None:
        return {}
    messages = build_flex_distill_messages(
        question,
        correctness=correctness,
        reasoning_summary=_clean_flex_text(str(result.get("reasoning_summary") or "")),
        trajectory_highlights=_format_flex_trajectory(result.get("trajectory") or []),
    )
    payload, _raw, _usage = llm.chat_json(messages, max_tokens=500, temperature=0.0)
    return payload


def _default_flex_distillation(
    question: QuestionRecord,
    result: dict[str, Any],
    *,
    correctness: bool,
) -> dict[str, Any]:
    zone = "golden" if correctness else "warning"
    reasoning = " ".join(str(result.get("reasoning_summary") or "").split())
    return {
        "strategy": {
            "title": _default_flex_title(question, level="strategy", correctness=correctness),
            "summary": _default_summary(question, "strategy", zone),
            "content": reasoning or _default_content(result, zone),
        },
        "pattern": {
            "title": _default_flex_title(question, level="pattern", correctness=correctness),
            "summary": _default_summary(question, "pattern", zone),
            "content": _default_content(result, zone),
        },
        "case": {
            "title": _default_flex_title(question, level="case", correctness=correctness),
            "summary": _default_summary(question, "case", zone),
            "content": reasoning or _default_content(result, zone),
        },
    }


def _default_summary(question: QuestionRecord, level: str, zone: str) -> str:
    return f"{zone} {level} distilled from a resolved {question['domain']} forecasting example."


def _default_content(result: dict[str, Any], zone: str) -> str:
    summary = " ".join(str(result.get("reasoning_summary") or "").split())
    if summary:
        return summary
    if zone == "golden":
        return "Preserve the successful evidence-weighting pattern and reuse it on similar questions."
    return "Preserve this warning so similar questions avoid repeating the same mistake."


def _sanitize_flex_payload(
    payload: dict[str, Any],
    *,
    question: QuestionRecord,
    correctness: bool,
    result: dict[str, Any],
) -> dict[str, dict[str, str]]:
    sanitized: dict[str, dict[str, str]] = {}
    for level in ("strategy", "pattern", "case"):
        raw_block = payload.get(level)
        block = raw_block if isinstance(raw_block, dict) else {}
        title = _clean_flex_text(str(block.get("title") or ""))
        summary = _clean_flex_text(str(block.get("summary") or ""))
        content = _clean_flex_text(str(block.get("content") or ""))
        if not title or _looks_generic_flex_title(title):
            title = _default_flex_title(question, level=level, correctness=correctness)
        if not summary:
            summary = _default_summary(question, level, "golden" if correctness else "warning")
        if not content:
            content = _default_content(result, "golden" if correctness else "warning")
        sanitized[level] = {
            "title": title,
            "summary": summary,
            "content": content,
        }
    return sanitized


def _clean_flex_text(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    fenced = _extract_fenced_json(text)
    if fenced:
        text = fenced
    parsed = _parse_json_like_reasoning(text)
    if parsed:
        text = parsed
    text = re.sub(r"\bmarket[_ ]?id\b\s*[:=]\s*\S+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bpredicted_prob(?:ability)?\b\s*[:=]\s*[-+]?[0-9.]+", "", text, flags=re.IGNORECASE)
    text = " ".join(text.replace("```", " ").split())
    return text[:600].strip(" ,;:-")


def _format_flex_trajectory(trajectory: list[dict[str, Any]]) -> str:
    if not trajectory:
        return "(none)"
    lines: list[str] = []
    for step in trajectory[-10:]:
        step_name = str(step.get("step") or "step")
        if step_name.startswith("tool_call"):
            lines.append(
                f"- {step_name}: called {step.get('tool_name')} with {str(step.get('arguments') or '')[:180]}"
            )
            continue
        if step_name.startswith("search_result"):
            hits = step.get("hits") if isinstance(step.get("hits"), list) else []
            titles = ", ".join(str(hit.get("title") or "") for hit in hits[:2] if isinstance(hit, dict))
            warning = str(step.get("warning") or "").strip()
            note = warning or titles or "no useful hits"
            lines.append(f"- {step_name}: search outcome -> {note}")
            continue
        if step_name.startswith("memory_result"):
            hits = step.get("hits") if isinstance(step.get("hits"), list) else []
            cues = ", ".join(str(hit.get("title") or "") for hit in hits[:2] if isinstance(hit, dict))
            lines.append(f"- {step_name}: memory retrieval -> {cues or 'no useful memories'}")
            continue
        if step_name.startswith("openbb_result"):
            lines.append(f"- {step_name}: openbb preview -> {json.dumps(step, ensure_ascii=False)[:220]}")
            continue
        if step_name.startswith("code_interpreter_result"):
            lines.append(f"- {step_name}: code output -> {_clean_flex_text(str(step.get('content') or ''))[:220]}")
            continue
        if step_name == "assistant":
            lines.append(f"- assistant: {_clean_flex_text(str(step.get('content') or ''))[:220]}")
    return "\n".join(lines)


def _default_flex_title(question: QuestionRecord, *, level: str, correctness: bool) -> str:
    task_shape = _infer_task_shape(question)
    if level == "strategy":
        return "Prioritize task-specific evidence before base-rate extrapolation" if correctness else "Do not confuse missing confirmation with low event probability"
    if level == "pattern":
        return "Sequence targeted evidence checks before numeric estimation" if correctness else "Stop broad search when results stay off-topic and switch approach"
    if level == "case":
        return task_shape
    return f"{question['domain']} experience"


def _infer_task_shape(question: QuestionRecord) -> str:
    title = " ".join(str(question.get("question") or "").split())
    title = re.sub(r"\bby\s+[A-Z][a-z]+\s+\d{1,2}\b.*$", "", title, flags=re.IGNORECASE)
    title = re.sub(r"\s+", " ", title).strip(" ?")
    return title[:90] or f"{question['domain']} forecasting case"


def _looks_generic_flex_title(text: str) -> bool:
    lowered = " ".join(text.lower().split())
    generic_tokens = {
        "golden strategy",
        "golden pattern",
        "golden case",
        "warning strategy",
        "warning pattern",
        "warning case",
        "case note",
    }
    if lowered in generic_tokens:
        return True
    if lowered.endswith("case note"):
        return True
    return False


def _extract_fenced_json(text: str) -> str:
    marker = "```json"
    lowered = text.lower()
    start = lowered.find(marker)
    if start < 0:
        return ""
    remainder = text[start + len(marker) :]
    end = remainder.find("```")
    if end < 0:
        return remainder.strip()
    return remainder[:end].strip()


def _parse_json_like_reasoning(text: str) -> str:
    text = text.strip()
    if not text.startswith("{"):
        return ""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return ""
    if not isinstance(payload, dict):
        return ""
    reasoning = str(payload.get("reasoning_summary") or payload.get("summary") or "").strip()
    return reasoning


def _is_correct_prediction(question: QuestionRecord, result: dict[str, Any]) -> bool:
    predicted_prob = float(result.get("predicted_prob") or 0.5)
    return (predicted_prob >= 0.5) == bool(int(question["label"]))
