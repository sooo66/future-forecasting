"""Direct IO baseline forecasting method."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from forecasting.contracts import ForecastMethod, MethodArtifact, MethodRuntimeContext, MethodSession, QuestionRecord
from forecasting.methods._agentic_shared import coerce_config, run_direct_io_forecast


@dataclass(frozen=True)
class DirectIOConfig:
    max_tokens: int = 300
    temperature: float = 0.0


class DirectIOMethod(ForecastMethod):
    method_id = "direct_io"

    def build_session(self, runtime_ctx: MethodRuntimeContext, method_config: Any | None = None) -> MethodSession:
        return _DirectIOSession(runtime_ctx.make_llm(), coerce_config(method_config, DirectIOConfig))


class _DirectIOSession(MethodSession):
    def __init__(self, llm: Any, config: DirectIOConfig) -> None:
        self._llm = llm
        self._config = config

    def run_question(self, question: QuestionRecord):
        return run_direct_io_forecast(
            question,
            self._llm,
            method_name="direct_io",
            max_tokens=self._config.max_tokens,
            temperature=self._config.temperature,
        )

    def finalize(self) -> list[MethodArtifact]:
        return []
