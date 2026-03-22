"""BM25 retrieval-augmented forecasting method."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from forecasting.contracts import ForecastMethod, MethodArtifact, MethodRuntimeContext, MethodSession, QuestionRecord
from forecasting.methods._agentic_shared import coerce_config, run_bm25_rag_forecast


@dataclass(frozen=True)
class Bm25RagConfig:
    search_top_k: int = 3
    search_content_chars: int = 512
    rag_max_per_source_type: int = 2
    max_tokens: int = 350
    temperature: float = 0.0


class Bm25RagMethod(ForecastMethod):
    method_id = "bm25_rag"

    def build_session(self, runtime_ctx: MethodRuntimeContext, method_config: Any | None = None) -> MethodSession:
        return _Bm25RagSession(runtime_ctx, coerce_config(method_config, Bm25RagConfig))


class _Bm25RagSession(MethodSession):
    def __init__(self, runtime_ctx: MethodRuntimeContext, config: Bm25RagConfig) -> None:
        self._runtime_ctx = runtime_ctx
        self._config = config
        self._llm = runtime_ctx.make_llm()

    def run_question(self, question: QuestionRecord):
        return run_bm25_rag_forecast(
            question,
            self._llm,
            self._runtime_ctx.search_engine,
            method_name="bm25_rag",
            search_top_k=self._config.search_top_k,
            search_content_chars=self._config.search_content_chars,
            rag_max_per_source_type=self._config.rag_max_per_source_type,
            max_tokens=self._config.max_tokens,
            temperature=self._config.temperature,
        )

    def finalize(self) -> list[MethodArtifact]:
        return []
