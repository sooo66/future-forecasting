"""OpenBB helpers shared by agent and forecasting."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
import inspect
import json
import os
import threading
import time
from typing import Any, Callable, Optional

from openbb import obb


_PROXY_ENV_KEYS = [
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "all_proxy",
    "ALL_PROXY",
]
_TIINGO_MIN_INTERVAL_SECONDS = float(os.getenv("TIINGO_MIN_INTERVAL_SECONDS", "1.0") or 1.0)
_TIINGO_RATE_LIMIT_COOLDOWN_SECONDS = float(os.getenv("TIINGO_RATE_LIMIT_COOLDOWN_SECONDS", "60") or 60.0)
_TIINGO_CALL_LOCK = threading.Lock()
_TIINGO_LAST_CALL_AT = 0.0
_TIINGO_COOLDOWN_UNTIL = 0.0
_OPENBB_RESULT_CACHE: dict[str, dict[str, Any]] = {}


@dataclass(frozen=True)
class OpenBBFunctionSpec:
    path: str
    description: str
    default_provider: Optional[str]
    example_params: dict[str, Any]


SAFE_OPENBB_FUNCTIONS: dict[str, OpenBBFunctionSpec] = {
    # === Equity (Stocks) ===
    "equity.price.quote": OpenBBFunctionSpec(
        path="equity.price.quote",
        description="股票最新报价。provider: fmp（需 FMP_API_KEY），不支持 tiingo。",
        default_provider="fmp",
        example_params={"symbol": "AAPL"},
    ),
    "equity.price.historical": OpenBBFunctionSpec(
        path="equity.price.historical",
        description="股票历史价格。provider: tiingo（需 TIINGO_API_KEY），可选 fmp/polygon。",
        default_provider="tiingo",
        example_params={"symbol": "AAPL", "start_date": "2026-02-01", "end_date": "2026-02-10"},
    ),
    "equity.profile": OpenBBFunctionSpec(
        path="equity.profile",
        description="股票公司简介与基本信息。provider: fmp（需 FMP_API_KEY）。",
        default_provider="fmp",
        example_params={"symbol": "AAPL"},
    ),
    "equity.ownership.institutional": OpenBBFunctionSpec(
        path="equity.ownership.institutional",
        description="机构持仓（主要股东、机构持有比例）。provider: fmp。",
        default_provider="fmp",
        example_params={"symbol": "AAPL"},
    ),
    "equity.fundamental.metrics": OpenBBFunctionSpec(
        path="equity.fundamental.metrics",
        description="估值指标（P/E、EV/EBITDA、PEG 等）。provider: fmp。",
        default_provider="fmp",
        example_params={"symbol": "AAPL"},
    ),
    "equity.fundamental.income": OpenBBFunctionSpec(
        path="equity.fundamental.income",
        description="损益表（收入、利润等）。provider: fmp 或 polygon。",
        default_provider="fmp",
        example_params={"symbol": "AAPL", "period": "annual", "limit": 5},
    ),
    "equity.fundamental.balance": OpenBBFunctionSpec(
        path="equity.fundamental.balance",
        description="资产负债表。provider: fmp 或 polygon。",
        default_provider="fmp",
        example_params={"symbol": "AAPL", "period": "annual", "limit": 5},
    ),
    "equity.fundamental.cash": OpenBBFunctionSpec(
        path="equity.fundamental.cash",
        description="现金流量表。provider: fmp 或 polygon。",
        default_provider="fmp",
        example_params={"symbol": "AAPL", "period": "annual", "limit": 5},
    ),
    "equity.fundamental.dividends": OpenBBFunctionSpec(
        path="equity.fundamental.dividends",
        description="分红历史。provider: fmp。",
        default_provider="fmp",
        example_params={"symbol": "AAPL", "limit": 10},
    ),
    "equity.fundamental.historical_splits": OpenBBFunctionSpec(
        path="equity.fundamental.historical_splits",
        description="股票拆分历史。provider: fmp。",
        default_provider="fmp",
        example_params={"symbol": "AAPL"},
    ),
    "equity.fundamental.historical_eps": OpenBBFunctionSpec(
        path="equity.fundamental.historical_eps",
        description="历史 EPS 数据。provider: fmp。",
        default_provider="fmp",
        example_params={"symbol": "AAPL"},
    ),
    # === Crypto - tiingo ===
    "crypto.price.historical": OpenBBFunctionSpec(
        path="crypto.price.historical",
        description="加密货币历史价格。provider: tiingo（需 TIINGO_API_KEY）。",
        default_provider="tiingo",
        example_params={"symbol": "BTC-USD", "start_date": "2026-02-01", "end_date": "2026-02-10"},
    ),
    # === Currency/Forex - tiingo ===
    "currency.price.historical": OpenBBFunctionSpec(
        path="currency.price.historical",
        description="外汇历史价格，symbol 格式 USD-CNY。provider: tiingo（需 TIINGO_API_KEY）。",
        default_provider="tiingo",
        example_params={"symbol": "USD-CNY", "start_date": "2026-02-01", "end_date": "2026-02-10"},
    ),
    # === Index ===
    "index.available": OpenBBFunctionSpec(
        path="index.available",
        description="可用指数列表。provider: fmp 或 yfinance。",
        default_provider="fmp",
        example_params={},
    ),
    "index.price.historical": OpenBBFunctionSpec(
        path="index.price.historical",
        description="指数历史价格。provider: fmp 或 polygon。",
        default_provider="fmp",
        example_params={"symbol": "^GSPC", "start_date": "2026-02-01", "end_date": "2026-02-10"},
    ),
    # === Macroeconomic - FRED / OECD ===
    "economy.cpi": OpenBBFunctionSpec(
        path="economy.cpi",
        description="消费者价格指数 (CPI) 通胀数据。provider: fred（需 FRED_API_KEY）。",
        default_provider="fred",
        example_params={"year": 2025, "month": 1},
    ),
    "economy.unemployment": OpenBBFunctionSpec(
        path="economy.unemployment",
        description="失业率。provider: oecd。",
        default_provider="oecd",
        example_params={"country": "USA"},
    ),
    "economy.gdp.real": OpenBBFunctionSpec(
        path="economy.gdp.real",
        description="实际 GDP。provider: oecd 或 econdb。",
        default_provider="oecd",
        example_params={"country": "USA"},
    ),
    "economy.gdp.nominal": OpenBBFunctionSpec(
        path="economy.gdp.nominal",
        description="名义 GDP。provider: oecd 或 econdb。",
        default_provider="oecd",
        example_params={"country": "USA"},
    ),
    "economy.interest_rates": OpenBBFunctionSpec(
        path="economy.interest_rates",
        description="基准利率。provider: oecd。",
        default_provider="oecd",
        example_params={"country": "USA"},
    ),
    "economy.balance_of_payments": OpenBBFunctionSpec(
        path="economy.balance_of_payments",
        description="国际收支平衡表。provider: fred（需 FRED_API_KEY）。",
        default_provider="fred",
        example_params={},
    ),
    "economy.central_bank_holdings": OpenBBFunctionSpec(
        path="economy.central_bank_holdings",
        description="央行资产持有量。provider: federal_reserve。",
        default_provider="federal_reserve",
        example_params={},
    ),
    "economy.indicators": OpenBBFunctionSpec(
        path="economy.indicators",
        description="宏观经济领先/滞后指标（PMI、消费者信心等）。provider: econdb 或 imf。",
        default_provider="econdb",
        example_params={"name": "Composite Leading Indicator"},
    ),
    # === Fixed Income / Treasury ===
    "fixedincome.government.treasury_rates": OpenBBFunctionSpec(
        path="fixedincome.government.treasury_rates",
        description="美国国债收益率曲线。provider: federal_reserve 或 fmp。",
        default_provider="federal_reserve",
        example_params={"start_date": "2026-02-01", "end_date": "2026-02-10"},
    ),
}


def list_supported_openbb_functions() -> list[dict[str, Any]]:
    items = []
    for key, spec in SAFE_OPENBB_FUNCTIONS.items():
        target = _resolve_openbb_callable(key)
        items.append(
            {
                "function": key,
                "description": spec.description,
                "default_provider": spec.default_provider,
                "signature": str(inspect.signature(target)),
                "example_params": _sanitize(spec.example_params),
            }
        )
    return items


def call_openbb_function(
    function: str,
    *,
    params: Optional[dict[str, Any]] = None,
    limit: int | None = None,
    invoker: Optional[Callable[[str, dict[str, Any]], Any]] = None,
) -> dict[str, Any]:
    raw_function = str(function or "").strip()
    if raw_function.lower() in {"", "list", "help"}:
        return {
            "supported_functions": list_supported_openbb_functions(),
            "notes": (
                "Provider 说明：tiingo（需 TIINGO_API_KEY）用于历史价格；fmp（需 FMP_API_KEY）用于报价/基本面；"
                "fred（需 FRED_API_KEY）用于宏观经济；oecd 用于 OECD 经济数据；federal_reserve 用于央行/国债数据。"
            ),
        }

    if raw_function not in SAFE_OPENBB_FUNCTIONS:
        return {
            "error": f"Unsupported OpenBB function: {raw_function}",
            "supported_functions": [item["function"] for item in list_supported_openbb_functions()],
        }

    if params is not None and not isinstance(params, dict):
        return {"error": "openbb.params must be a JSON object"}
    payload = dict(params or {})

    spec = SAFE_OPENBB_FUNCTIONS[raw_function]
    if spec.default_provider and "provider" not in payload:
        payload["provider"] = spec.default_provider
    cache_key = _make_openbb_cache_key(raw_function, payload)
    cached_payload = _OPENBB_RESULT_CACHE.get(cache_key)
    if cached_payload is not None:
        return {**cached_payload, "cache_hit": True}

    invoke = invoker or _invoke_openbb
    try:
        result = _invoke_with_retry(raw_function, payload, invoke=invoke)
    except Exception as exc:
        cooldown_remaining = _tiingo_cooldown_remaining(payload)
        error_payload = {
            "function": raw_function,
            "params": _sanitize(payload),
            "error": str(exc),
        }
        if _is_openbb_rate_limit_error(exc):
            error_payload["retryable"] = True
            error_payload["provider"] = payload.get("provider")
            error_payload["error_type"] = "rate_limit"
            if cooldown_remaining > 0:
                error_payload["cooldown_seconds"] = cooldown_remaining
            if cached_payload is not None:
                return {
                    **cached_payload,
                    "cache_hit": True,
                    "warning": "Returned cached OpenBB result after provider rate limit.",
                    "stale_cache": True,
                }
        return {
            **error_payload,
        }

    normalized = _coerce_result(result)
    if isinstance(normalized, list):
        result_count = len(normalized)
        if limit is None:
            returned = normalized
        else:
            result_limit = max(1, min(int(limit or 20), 50))
            returned = normalized[:result_limit]
    else:
        returned = normalized
        result_count = None

    response_payload = {
        "function": raw_function,
        "params": _sanitize(payload),
        "result_count": result_count,
        "results": returned,
    }
    _OPENBB_RESULT_CACHE[cache_key] = response_payload
    return response_payload


def _invoke_openbb(function: str, params: dict[str, Any]) -> Any:
    target = _resolve_openbb_callable(function)
    with _without_proxy_env():
        return target(**params)


def _invoke_with_retry(
    function: str,
    params: dict[str, Any],
    *,
    invoke: Callable[[str, dict[str, Any]], Any],
) -> Any:
    delays = [0.0, 1.0, 2.0]
    last_exc: Exception | None = None
    for index, delay in enumerate(delays):
        if delay > 0:
            time.sleep(delay)
        try:
            _guard_tiingo_request(params)
            return invoke(function, params)
        except Exception as exc:
            last_exc = exc
            _record_tiingo_result(params, exc)
            if not _is_openbb_rate_limit_error(exc) or index == len(delays) - 1:
                raise
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"Failed to invoke OpenBB function: {function}")


def _is_openbb_rate_limit_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "too many requests" in message
        or "rate limit" in message
        or "ratelimit" in message
        or "429" in message
        or "tiingo" in message and "limit" in message
    )


def _resolve_openbb_callable(function: str) -> Callable[..., Any]:
    current: Any = obb
    for part in function.split("."):
        current = getattr(current, part)
    if not callable(current):
        raise TypeError(f"OpenBB path is not callable: {function}")
    return current


def _coerce_result(result: Any) -> Any:
    if hasattr(result, "to_dict"):
        try:
            return _sanitize(result.to_dict(orient="records"))
        except TypeError:
            return _sanitize(result.to_dict())
    return _sanitize(result)


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize(item) for item in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    return value


def _make_openbb_cache_key(function: str, params: dict[str, Any]) -> str:
    payload = {
        "function": function,
        "params": _sanitize(params),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _uses_tiingo(params: dict[str, Any]) -> bool:
    provider = str(params.get("provider") or "").strip().lower()
    return provider == "tiingo"


def _guard_tiingo_request(params: dict[str, Any]) -> None:
    if not _uses_tiingo(params):
        return
    global _TIINGO_LAST_CALL_AT
    with _TIINGO_CALL_LOCK:
        now = time.monotonic()
        if _TIINGO_COOLDOWN_UNTIL > now:
            remaining = _TIINGO_COOLDOWN_UNTIL - now
            raise RuntimeError(f"tiingo cooldown active for {remaining:.1f}s after rate limiting")
        wait_for = (_TIINGO_LAST_CALL_AT + _TIINGO_MIN_INTERVAL_SECONDS) - now
        if wait_for > 0:
            time.sleep(wait_for)
        _TIINGO_LAST_CALL_AT = time.monotonic()


def _record_tiingo_result(params: dict[str, Any], exc: Exception | None) -> None:
    if not _uses_tiingo(params):
        return
    global _TIINGO_COOLDOWN_UNTIL
    if exc is None:
        return
    if _is_openbb_rate_limit_error(exc):
        with _TIINGO_CALL_LOCK:
            _TIINGO_COOLDOWN_UNTIL = max(
                _TIINGO_COOLDOWN_UNTIL,
                time.monotonic() + _TIINGO_RATE_LIMIT_COOLDOWN_SECONDS,
            )


def _tiingo_cooldown_remaining(params: dict[str, Any]) -> int | None:
    if not _uses_tiingo(params):
        return None
    remaining = int(max(0.0, _TIINGO_COOLDOWN_UNTIL - time.monotonic()))
    return remaining or None


@contextmanager
def _without_proxy_env():
    old_values = {key: os.environ.get(key) for key in _PROXY_ENV_KEYS}
    for key in _PROXY_ENV_KEYS:
        os.environ.pop(key, None)
    try:
        yield
    finally:
        for key, value in old_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
