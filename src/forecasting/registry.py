"""Registries for forecasting methods."""

from __future__ import annotations

from forecasting.contracts import ForecastMethod
from forecasting.methods import (
    AgenticNoMemoryMethod,
    Bm25RagMethod,
    DirectIOMethod,
    FlexMethod,
    ReasoningBankMethod,
)


_METHODS: dict[str, ForecastMethod] = {
    method.method_id: method
    for method in [
        DirectIOMethod(),
        Bm25RagMethod(),
        AgenticNoMemoryMethod(),
        ReasoningBankMethod(),
        FlexMethod(),
    ]
}


def list_methods() -> list[str]:
    return sorted(_METHODS)


def get_method(method_id: str) -> ForecastMethod:
    try:
        return _METHODS[method_id]
    except KeyError as exc:
        raise KeyError(f"Unknown forecasting method: {method_id}") from exc
