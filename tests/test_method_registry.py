from __future__ import annotations

from forecasting.registry import get_method, list_methods


def test_method_registry_lists_expected_methods():
    methods = list_methods()
    assert methods == sorted(["agentic_nomem", "bm25_rag", "direct_io", "flex", "reasoningbank"])


def test_method_registry_resolves_named_methods():
    assert get_method("direct_io").method_id == "direct_io"
    assert get_method("reasoningbank").method_id == "reasoningbank"
    assert get_method("flex").method_id == "flex"
