"""ReasoningBank-style forecasting method."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from forecasting.contracts import ForecastMethod, MethodArtifact, MethodRuntimeContext, MethodSession, QuestionRecord
from forecasting.embeddings import DEFAULT_EMBEDDING_MODEL, build_text_embedder
from forecasting.memory import MemoryItem, ReasoningBankRecord, ReasoningBankStore
from forecasting.methods._agentic_shared import build_memory_query, coerce_config, run_agentic_forecast
from forecasting.prompts import (
    build_reasoningbank_extraction_messages,
    format_reasoningbank_trajectory,
)
from forecasting.question_tools import ResidentCodeInterpreterTool


@dataclass(frozen=True)
class ReasoningBankConfig:
    agent_max_steps: int = 5
    top_k: int = 1
    search_top_k: int = 3
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL
    embedding_device: str | None = None
    artifact_filename: str = "reasoningbank_mem.jsonl"


class ReasoningBankMethod(ForecastMethod):
    method_id = "reasoningbank"

    def build_session(self, runtime_ctx: MethodRuntimeContext, method_config: Any | None = None) -> MethodSession:
        return _ReasoningBankSession(runtime_ctx, coerce_config(method_config, ReasoningBankConfig))


class _ReasoningBankSession(MethodSession):
    def __init__(self, runtime_ctx: MethodRuntimeContext, config: ReasoningBankConfig) -> None:
        self._runtime_ctx = runtime_ctx
        self._config = config
        self._llm = runtime_ctx.make_llm()
        self._store = ReasoningBankStore(
            embedder=build_text_embedder(
                model_name=config.embedding_model_name,
                device=config.embedding_device,
            ),
            model_name=config.embedding_model_name,
        )
        self._code_interpreter = ResidentCodeInterpreterTool(
            work_dir=runtime_ctx.project_root / ".qwen_agent_workspace" / "forecasting" / "reasoningbank"
        )

    def run_question(self, question: QuestionRecord):
        memories = self._store.retrieve(
            build_memory_query(question),
            open_time=question["open_time"],
            top_k=self._config.top_k,
        )
        result = run_agentic_forecast(
            question,
            self._llm,
            self._runtime_ctx.search_engine,
            project_root=self._runtime_ctx.project_root,
            method_name="reasoningbank",
            agent_max_steps=self._config.agent_max_steps,
            search_top_k=self._config.search_top_k,
            injected_memories=memories,
            code_interpreter_tool=self._code_interpreter,
        )
        if "error" not in result:
            self._store.queue(_build_reasoningbank_record(question, result, llm=self._llm))
        return result

    def finalize(self) -> list[MethodArtifact]:
        return [
            MethodArtifact(
                filename=self._config.artifact_filename,
                format="jsonl",
                payload=self._store.artifact_rows(),
            ),
        ]


def _build_reasoningbank_record(
    question: QuestionRecord,
    result: dict[str, Any],
    *,
    llm: Any | None = None,
) -> ReasoningBankRecord:
    success_or_failure = "success" if _is_correct_prediction(question, result) else "failure"
    memory_items = _extract_reasoningbank_items(question, result, success_or_failure=success_or_failure, llm=llm)
    if not memory_items:
        memory_items = [_default_memory_item(question, result, success_or_failure=success_or_failure)]
    return ReasoningBankRecord(
        record_id=f"reasoningbank-{question['market_id']}",
        source_question_id=question["market_id"],
        query=build_memory_query(question),
        trajectory=list(result.get("trajectory") or []),
        memory_items=memory_items[:3],
        created_at=question["resolve_time"],
        source_open_time=question["open_time"],
        source_resolved_time=question["resolve_time"],
        outcome=int(question["label"]),
        predicted_prob=float(result.get("predicted_prob") or 0.5),
        success_or_failure=success_or_failure,
    )


def _extract_reasoningbank_items(
    question: QuestionRecord,
    result: dict[str, Any],
    *,
    success_or_failure: str,
    llm: Any | None,
) -> list[MemoryItem]:
    if llm is None:
        return []
    raw_text, _usage = llm.chat(
        build_reasoningbank_extraction_messages(
            query=question["question"],
            trajectory_text=format_reasoningbank_trajectory(result),
            success_or_failure=success_or_failure,
        ),
        max_tokens=700,
        temperature=0.0,
    )
    items = _parse_memory_items(raw_text)
    return items[:3]


def _parse_memory_items(text: str) -> list[MemoryItem]:
    blocks = re.split(r"\n(?=# Memory Item\b)", text or "")
    items: list[MemoryItem] = []
    for block in blocks:
        if "# Memory Item" not in block:
            continue
        title = _match_section(block, "Title")
        description = _match_section(block, "Description")
        content = _match_section(block, "Content")
        if not title or not description or not content:
            continue
        items.append(
            MemoryItem(
                title=title,
                description=description,
                content=content,
            )
        )
    return items


def _match_section(block: str, heading: str) -> str:
    pattern = rf"## {heading}\s*(.*?)(?=\n## |\Z)"
    match = re.search(pattern, block, flags=re.DOTALL)
    if not match:
        return ""
    value = match.group(1).strip()
    value = value.strip("`")
    return " ".join(value.split())


def _default_memory_item(
    question: QuestionRecord,
    result: dict[str, Any],
    *,
    success_or_failure: str,
) -> MemoryItem:
    label = int(question["label"])
    outcome = "YES" if label else "NO"
    summary = " ".join(str(result.get("reasoning_summary") or "").split())
    if success_or_failure == "success":
        title = f"{question['domain']} successful reasoning pattern"
        description = f"Resolved {outcome} correctly using reusable reasoning cues."
        content = summary or "Use cutoff-safe evidence, update from concrete signals, and finish with explicit uncertainty."
    else:
        title = f"{question['domain']} failure warning"
        description = f"Resolved {outcome}; preserve this as a warning for similar tasks."
        content = summary or "Record which cues were over- or under-weighted and avoid repeating the same mistake."
    return MemoryItem(title=title, description=description, content=content)


def _is_correct_prediction(question: QuestionRecord, result: dict[str, Any]) -> bool:
    predicted_prob = float(result.get("predicted_prob") or 0.5)
    return (predicted_prob >= 0.5) == bool(int(question["label"]))
