"""Core contracts for reusable forecasting methods and runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Protocol, TypedDict


class QuestionRecord(TypedDict, total=False):
    market_id: str
    question: str
    description: str
    resolution_criteria: str
    domain: str
    open_time: str
    resolve_time: str
    resolved_time: str
    sample_time: str
    difficulty: str
    sampled_prob_yes: float
    label: int
    horizon: str


class ForecastResult(TypedDict, total=False):
    market_id: str
    domain: str
    difficulty: str
    predicted_prob: float
    label: int
    brier_score: float
    trajectory: list[dict[str, Any]]
    reasoning_summary: str
    latency_sec: float
    total_tokens: int | None
    steps_count: int | None
    error: str


@dataclass(frozen=True)
class MethodArtifact:
    filename: str
    format: Literal["json", "jsonl"]
    payload: Any


@dataclass(frozen=True)
class MethodRuntimeContext:
    project_root: Path
    output_dir: Path
    search_engine: Any
    llm_factory: Callable[[], Any]

    def make_llm(self) -> Any:
        return self.llm_factory()


class MethodSession(Protocol):
    def run_question(self, question: QuestionRecord) -> ForecastResult: ...

    def finalize(self) -> list[MethodArtifact]: ...


class ForecastMethod(Protocol):
    method_id: str

    def build_session(
        self, runtime_ctx: MethodRuntimeContext, method_config: Any | None = None
    ) -> MethodSession: ...
