"""Simplified FLEX-style forecasting method."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from forecasting.contracts import ForecastMethod, MethodArtifact, MethodRuntimeContext, MethodSession, QuestionRecord
from forecasting.embeddings import DEFAULT_EMBEDDING_MODEL, build_text_embedder
from forecasting.memory import FlexExperience, FlexLibrary
from forecasting.methods._agentic_shared import build_memory_query, coerce_config, run_agentic_forecast
from forecasting.question_tools import FlexMemoryTool, ResidentCodeInterpreterTool

_SUCCESS_DISTILL_PROMPT = """You are a professional expert analyzing a solver's successful problem-solving process.
Your goal is to extract a generalizable recipe for success from this experience and organize it into three levels:
1. strategy: high-level, reusable guidance for similar problems.
2. pattern: mid-level reasoning template or methodological pattern.
3. case: concrete case-level insight from this task.

Return JSON only with keys strategy, pattern, case.
Each value must be an object with keys title, summary, content."""

_FAILURE_DISTILL_PROMPT = """You are a professional expert diagnosing a solver's failed problem-solving process.
Your goal is to identify reusable warnings and corrective guidance from this experience and organize it into three levels:
1. strategy: high-level warning or corrective guidance for similar problems.
2. pattern: mid-level failure pattern or corrective template.
3. case: concrete case-level warning from this task.

Return JSON only with keys strategy, pattern, case.
Each value must be an object with keys title, summary, content."""


@dataclass(frozen=True)
class FlexConfig:
    agent_max_steps: int = 7
    search_top_k: int = 3
    strategy_top_k: int = 5
    pattern_top_k: int = 5
    case_top_k: int = 5
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
        preloaded = self._library.retrieve_default_bundle(
            build_memory_query(question),
            open_time=question["open_time"],
            per_level={
                "strategy": self._config.strategy_top_k,
                "pattern": self._config.pattern_top_k,
                "case": self._config.case_top_k,
            },
        )
        result = run_agentic_forecast(
            question,
            self._llm,
            self._runtime_ctx.search_engine,
            project_root=self._runtime_ctx.project_root,
            method_name="flex",
            agent_max_steps=self._config.agent_max_steps,
            search_top_k=self._config.search_top_k,
            flex_memory_tool=FlexMemoryTool(self._library, cutoff_time=question["open_time"]),
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
    items: list[FlexExperience] = []
    for level in ["strategy", "pattern", "case"]:
        block = payload.get(level) or {}
        items.append(
            FlexExperience(
                experience_id=f"flex-{zone}-{level}-{question['market_id']}",
                source_question_id=question["market_id"],
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
    messages = [
        {
            "role": "system",
            "content": _SUCCESS_DISTILL_PROMPT if correctness else _FAILURE_DISTILL_PROMPT,
        },
        {"role": "user", "content": _flex_context(question, result)},
    ]
    payload, _raw, _usage = llm.chat_json(messages, max_tokens=500, temperature=0.0)
    return payload


def _flex_context(question: QuestionRecord, result: dict[str, Any]) -> str:
    return (
        f"Original Query: {question['question']}\n"
        f"Description: {' '.join(question['description'].split())[:800]}\n"
        f"Resolution Criteria: {' '.join(question['resolution_criteria'].split())[:800]}\n"
        f"Ground Truth Label: {question['label']}\n"
        f"Predicted Probability: {result.get('predicted_prob')}\n"
        f"Reasoning Summary: {result.get('reasoning_summary')}\n"
        f"Trajectory: {json.dumps(result.get('trajectory') or [], ensure_ascii=False)}\n"
    )


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
            "title": f"{question['domain']} {zone} strategy",
            "summary": _default_summary(question, "strategy", zone),
            "content": reasoning or _default_content(result, zone),
        },
        "pattern": {
            "title": f"{question['domain']} {zone} pattern",
            "summary": _default_summary(question, "pattern", zone),
            "content": _default_content(result, zone),
        },
        "case": {
            "title": f"{question['domain']} {zone} case note",
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


def _is_correct_prediction(question: QuestionRecord, result: dict[str, Any]) -> bool:
    predicted_prob = float(result.get("predicted_prob") or 0.5)
    return (predicted_prob >= 0.5) == bool(int(question["label"]))
