from __future__ import annotations

from datetime import date

from tools.openbb import call_openbb_function, list_supported_openbb_functions


def test_openbb_list_contains_expected_functions():
    functions = list_supported_openbb_functions()
    names = {item["function"] for item in functions}
    assert "equity.price.quote" in names
    assert "index.price.historical" in names


def test_openbb_call_applies_default_provider_and_sanitizes_result():
    captured = {}

    def fake_invoker(function: str, params: dict[str, object]):
        captured["function"] = function
        captured["params"] = dict(params)
        return [{"date": date(2026, 2, 2), "close": 270.01}]

    result = call_openbb_function(
        "equity.price.quote",
        params={"symbol": "AAPL"},
        invoker=fake_invoker,
    )

    assert captured["function"] == "equity.price.quote"
    assert captured["params"]["provider"] == "yfinance"
    assert result["results"][0]["date"] == "2026-02-02"


def test_openbb_rejects_unsupported_function():
    result = call_openbb_function("economy.gdp", params={})
    assert "error" in result
    assert result["supported_functions"]


def test_openbb_call_does_not_truncate_results_by_default():
    def fake_invoker(_function: str, _params: dict[str, object]):
        return [{"date": date(2026, 2, day + 1), "close": float(day)} for day in range(25)]

    result = call_openbb_function(
        "equity.price.historical",
        params={"symbol": "AAPL"},
        invoker=fake_invoker,
    )

    assert result["result_count"] == 25
    assert len(result["results"]) == 25
