from __future__ import annotations

from datetime import date

from tools import openbb as openbb_module
from tools.openbb import call_openbb_function, list_supported_openbb_functions


def _reset_openbb_state() -> None:
    openbb_module._OPENBB_RESULT_CACHE.clear()
    openbb_module._TIINGO_COOLDOWN_UNTIL = 0.0
    openbb_module._TIINGO_LAST_CALL_AT = 0.0


def test_openbb_list_contains_expected_functions():
    functions = list_supported_openbb_functions()
    names = {item["function"] for item in functions}
    assert "equity.price.quote" in names
    assert "equity.price.historical" in names
    assert "crypto.price.historical" in names
    assert "economy.cpi" in names


def test_openbb_call_applies_default_provider_and_sanitizes_result():
    _reset_openbb_state()
    captured = {}

    def fake_invoker(function: str, params: dict[str, object]):
        captured["function"] = function
        captured["params"] = dict(params)
        return [{"date": date(2026, 2, 2), "close": 270.01}]

    result = call_openbb_function(
        "equity.price.historical",
        params={"symbol": "AAPL"},
        invoker=fake_invoker,
    )

    assert captured["function"] == "equity.price.historical"
    assert captured["params"]["provider"] == "tiingo"
    assert result["results"][0]["date"] == "2026-02-02"


def test_openbb_rejects_unsupported_function():
    result = call_openbb_function("equity.nonexistent", params={})
    assert "error" in result
    assert result["supported_functions"]


def test_openbb_call_does_not_truncate_results_by_default():
    _reset_openbb_state()

    def fake_invoker(_function: str, _params: dict[str, object]):
        return [{"date": date(2026, 2, day + 1), "close": float(day)} for day in range(25)]

    result = call_openbb_function(
        "equity.price.historical",
        params={"symbol": "AAPL"},
        invoker=fake_invoker,
    )

    assert result["result_count"] == 25
    assert len(result["results"]) == 25


def test_openbb_retries_and_returns_retryable_rate_limit_error(monkeypatch):
    _reset_openbb_state()
    calls = {"count": 0}

    monkeypatch.setattr("tools.openbb.time.sleep", lambda _seconds: None)

    def fake_invoker(_function: str, _params: dict[str, object]):
        calls["count"] += 1
        raise RuntimeError("tiingo rate limit exceeded: 429")

    result = call_openbb_function(
        "equity.price.historical",
        params={"symbol": "AAPL"},
        invoker=fake_invoker,
    )

    assert calls["count"] == 1
    assert result["retryable"] is True
    assert result["error_type"] == "rate_limit"
    assert result["provider"] == "tiingo"


def test_openbb_caches_identical_results(monkeypatch):
    _reset_openbb_state()
    monkeypatch.setattr("tools.openbb.time.sleep", lambda _seconds: None)
    calls = {"count": 0}

    def fake_invoker(_function: str, _params: dict[str, object]):
        calls["count"] += 1
        return [{"date": date(2026, 2, 2), "close": 270.01}]

    first = call_openbb_function(
        "equity.price.historical",
        params={"symbol": "AAPL"},
        invoker=fake_invoker,
    )
    second = call_openbb_function(
        "equity.price.historical",
        params={"symbol": "AAPL"},
        invoker=fake_invoker,
    )

    assert calls["count"] == 1
    assert "cache_hit" not in first
    assert second["cache_hit"] is True
    assert second["results"] == first["results"]
