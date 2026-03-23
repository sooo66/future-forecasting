"""Naive retrieval-augmented forecasting method."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from forecasting.contracts import ForecastMethod, MethodArtifact, MethodRuntimeContext, MethodSession, QuestionRecord
from forecasting.methods._agentic_shared import coerce_config, run_naive_rag_forecast


@dataclass(frozen=True)
class NaiveRagConfig:
    search_top_k: int = 3
    search_content_chars: int = 512
    rag_max_per_source_type: int = 2
    max_tokens: int = 350
    temperature: float = 0.0


class NaiveRagMethod(ForecastMethod):
    method_id = "naive_rag"

    def build_session(self, runtime_ctx: MethodRuntimeContext, method_config: Any | None = None) -> MethodSession:
        return _NaiveRagSession(runtime_ctx, coerce_config(method_config, NaiveRagConfig))


class _NaiveRagSession(MethodSession):
    def __init__(self, runtime_ctx: MethodRuntimeContext, config: NaiveRagConfig) -> None:
        self._runtime_ctx = runtime_ctx
        self._config = config
        self._llm = runtime_ctx.make_llm()
        self._resolved_method_name = _resolve_naive_rag_name(runtime_ctx.search_engine)

    def run_question(self, question: QuestionRecord):
        return run_naive_rag_forecast(
            question,
            self._llm,
            self._runtime_ctx.search_engine,
            method_name=self._resolved_method_name,
            search_top_k=self._config.search_top_k,
            search_content_chars=self._config.search_content_chars,
            rag_max_per_source_type=self._config.rag_max_per_source_type,
            max_tokens=self._config.max_tokens,
            temperature=self._config.temperature,
        )

    def finalize(self) -> list[MethodArtifact]:
        return []


def _resolve_naive_rag_name(search_engine: Any) -> str:
    try:
        health = search_engine.health()
    except Exception:
        health = {}
    backend = str((health or {}).get("backend") or "").strip().lower()
    if backend == "exa":
        return "exa_rag"
    mode = str(
        getattr(search_engine, "default_mode", None)
        or (health or {}).get("default_mode")
        or ""
    ).strip().lower()
    if mode == "dense":
        return "dense_rag"
    if mode == "hybrid":
        return "hybrid_rag"
    return "naive_rag"


# Backward-compatible aliases for older imports.
Bm25RagConfig = NaiveRagConfig
Bm25RagMethod = NaiveRagMethod
