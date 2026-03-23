"""Registries for forecasting methods."""

from __future__ import annotations

from forecasting.contracts import ForecastMethod
from forecasting.methods import (
    AgenticNoMemoryMethod,
    DirectIOMethod,
    FlexMethod,
    NaiveRagMethod,
    ReasoningBankMethod,
)


_METHODS: dict[str, ForecastMethod] = {
    method.method_id: method
    for method in [
        DirectIOMethod(),
        NaiveRagMethod(),
        AgenticNoMemoryMethod(),
        ReasoningBankMethod(),
        FlexMethod(),
    ]
}
_METHOD_ALIASES = {
    "bm25_rag": "naive_rag",
}


def list_methods() -> list[str]:
    return sorted(_METHODS)


def get_method(method_id: str) -> ForecastMethod:
    resolved_method_id = _METHOD_ALIASES.get(method_id, method_id)
    try:
        return _METHODS[resolved_method_id]
    except KeyError as exc:
        raise KeyError(f"Unknown forecasting method: {method_id}") from exc
