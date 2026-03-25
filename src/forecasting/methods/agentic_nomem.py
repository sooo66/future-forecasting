"""Agentic forecasting method without memory augmentation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from forecasting.contracts import ForecastMethod, MethodArtifact, MethodRuntimeContext, MethodSession, QuestionRecord
from forecasting.methods._agentic import run_agentic_forecast
from forecasting.methods._shared import coerce_config


@dataclass(frozen=True)
class AgenticNoMemoryConfig:
    agent_max_steps: int = 8
    search_top_k: int = 3


class AgenticNoMemoryMethod(ForecastMethod):
    method_id = "agentic_nomem"

    def build_session(self, runtime_ctx: MethodRuntimeContext, method_config: Any | None = None) -> MethodSession:
        return _AgenticNoMemorySession(runtime_ctx, coerce_config(method_config, AgenticNoMemoryConfig))


class _AgenticNoMemorySession(MethodSession):
    def __init__(self, runtime_ctx: MethodRuntimeContext, config: AgenticNoMemoryConfig) -> None:
        self._runtime_ctx = runtime_ctx
        self._config = config
        self._llm = runtime_ctx.make_llm()

    def run_question(self, question: QuestionRecord):
        return run_agentic_forecast(
            question,
            self._llm,
            self._runtime_ctx.search_engine,
            project_root=self._runtime_ctx.project_root,
            method_name="agentic_nomem",
            agent_max_steps=self._config.agent_max_steps,
            search_top_k=self._config.search_top_k,
        )

    def finalize(self) -> list[MethodArtifact]:
        return []
