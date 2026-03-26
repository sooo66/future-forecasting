"""ReasoningBank-style forecasting method."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from forecasting.contracts import ForecastMethod, MethodArtifact, MethodRuntimeContext, MethodSession, QuestionRecord
from forecasting.embeddings import DEFAULT_EMBEDDING_MODEL, build_text_embedder
from forecasting.memory import MemoryItem, ReasoningBankStore
from forecasting.methods._agentic import (
    build_memory_query,
    run_agentic_forecast,
)
from forecasting.methods._shared import (
    coerce_config,
)
from forecasting.prompts import (
    build_reasoningbank_extraction_messages,
    format_reasoningbank_trajectory,
)


@dataclass(frozen=True)
class ReasoningBankConfig:
    agent_max_steps: int = 8
    max_tokens: int = 2048
    top_k: int = 1
    search_top_k: int = 3
    success_only: bool = False
    domain_match: bool = False
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

    def run_question(self, question: QuestionRecord):
        memories = self._store.retrieve(
            build_memory_query(question),
            top_k=self._config.top_k,
        )
        result = run_agentic_forecast(
            question,
            self._llm,
            self._runtime_ctx.search_engine,
            project_root=self._runtime_ctx.project_root,
            method_name="reasoningbank",
            agent_max_steps=self._config.agent_max_steps,
            max_tokens=self._config.max_tokens,
            search_top_k=self._config.search_top_k,
            injected_memories=memories,
        )
        if "error" not in result:
            items = _extract_reasoningbank_items(question, result, llm=self._llm)
            if not items:
                items = [_default_memory_item(question, result)]
            self._store.add_items(items)
        return result

    def finalize(self) -> list[MethodArtifact]:
        return [
            MethodArtifact(
                filename=self._config.artifact_filename,
                format="jsonl",
                payload=self._store.artifact_rows(),
            ),
        ]


def _extract_reasoningbank_items(
    question: QuestionRecord,
    result: dict[str, Any],
    *,
    llm: Any | None,
) -> list[MemoryItem]:
    if llm is None:
        return []
    # Use LLM-as-judge to determine success/failure for extraction prompt selection
    predicted_prob = float(result.get("predicted_prob") or 0.5)
    success_or_failure = "success" if (predicted_prob >= 0.5) == bool(int(question.get("label", 0))) else "failure"
    raw_text, _usage = llm.chat(
        build_reasoningbank_extraction_messages(
            query=question["question"],
            trajectory_text=format_reasoningbank_trajectory(result),
            success_or_failure=success_or_failure,
        ),
        max_tokens=2048,
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
) -> MemoryItem:
    predicted_prob = float(result.get("predicted_prob") or 0.5)
    success_or_failure = "success" if (predicted_prob >= 0.5) == bool(int(question.get("label", 0))) else "failure"
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
