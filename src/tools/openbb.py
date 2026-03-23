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
from yfinance.exceptions import YFRateLimitError


_PROXY_ENV_KEYS = [
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "all_proxy",
    "ALL_PROXY",
]
_YFINANCE_MIN_INTERVAL_SECONDS = float(os.getenv("YFINANCE_MIN_INTERVAL_SECONDS", "1.5") or 1.5)
_YFINANCE_RATE_LIMIT_COOLDOWN_SECONDS = float(os.getenv("YFINANCE_RATE_LIMIT_COOLDOWN_SECONDS", "60") or 60.0)
_YFINANCE_CALL_LOCK = threading.Lock()
_YFINANCE_LAST_CALL_AT = 0.0
_YFINANCE_COOLDOWN_UNTIL = 0.0
_OPENBB_RESULT_CACHE: dict[str, dict[str, Any]] = {}


@dataclass(frozen=True)
class OpenBBFunctionSpec:
    path: str
    description: str
    default_provider: Optional[str]
    example_params: dict[str, Any]


SAFE_OPENBB_FUNCTIONS: dict[str, OpenBBFunctionSpec] = {
    "equity.price.quote": OpenBBFunctionSpec(
        path="equity.price.quote",
        description="股票最新报价，默认 provider 使用 yfinance。",
        default_provider="yfinance",
        example_params={"symbol": "AAPL"},
    ),
    "equity.price.historical": OpenBBFunctionSpec(
        path="equity.price.historical",
        description="股票历史价格，默认 provider 使用 yfinance。",
        default_provider="yfinance",
        example_params={"symbol": "AAPL", "start_date": "2026-02-01", "end_date": "2026-02-10"},
    ),
    "equity.profile": OpenBBFunctionSpec(
        path="equity.profile",
        description="股票公司简介与基本信息，默认 provider 使用 yfinance。",
        default_provider="yfinance",
        example_params={"symbol": "AAPL"},
    ),
    "crypto.price.historical": OpenBBFunctionSpec(
        path="crypto.price.historical",
        description="加密货币历史价格，默认 provider 使用 yfinance。",
        default_provider="yfinance",
        example_params={"symbol": "BTC-USD", "start_date": "2026-02-01", "end_date": "2026-02-10"},
    ),
    "currency.price.historical": OpenBBFunctionSpec(
        path="currency.price.historical",
        description="外汇历史价格，symbol 可写成 USD-CNY 这类格式，默认 provider 使用 yfinance。",
        default_provider="yfinance",
        example_params={"symbol": "USD-CNY", "start_date": "2026-02-01", "end_date": "2026-02-10"},
    ),
    "index.available": OpenBBFunctionSpec(
        path="index.available",
        description="可用指数列表，默认 provider 使用 yfinance。",
        default_provider="yfinance",
        example_params={},
    ),
    "index.price.historical": OpenBBFunctionSpec(
        path="index.price.historical",
        description="指数历史价格，默认 provider 使用 yfinance。",
        default_provider="yfinance",
        example_params={"symbol": "^GSPC", "start_date": "2026-02-01", "end_date": "2026-02-10"},
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
            "notes": "当前只开放首批无需额外 API key 的 OpenBB 函数；需要密钥的 provider 先不接入。",
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
        cooldown_remaining = _yfinance_cooldown_remaining(payload)
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
                    "warning": "Returned cached OpenBB result after yfinance rate limit.",
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
            _guard_yfinance_request(params)
            return invoke(function, params)
        except Exception as exc:
            last_exc = exc
            _record_yfinance_result(params, exc)
            if not _is_openbb_rate_limit_error(exc) or index == len(delays) - 1:
                raise
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"Failed to invoke OpenBB function: {function}")


def _is_openbb_rate_limit_error(exc: Exception) -> bool:
    if isinstance(exc, YFRateLimitError):
        return True
    message = str(exc).lower()
    return (
        "too many requests" in message
        or "rate limit" in message
        or "ratelimit" in message
        or "yfinance cooldown active" in message
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


def _uses_yfinance(params: dict[str, Any]) -> bool:
    return str(params.get("provider") or "").strip().lower() == "yfinance"


def _guard_yfinance_request(params: dict[str, Any]) -> None:
    if not _uses_yfinance(params):
        return
    global _YFINANCE_LAST_CALL_AT
    with _YFINANCE_CALL_LOCK:
        now = time.monotonic()
        if _YFINANCE_COOLDOWN_UNTIL > now:
            remaining = _YFINANCE_COOLDOWN_UNTIL - now
            raise RuntimeError(f"yfinance cooldown active for {remaining:.1f}s after rate limiting")
        wait_for = (_YFINANCE_LAST_CALL_AT + _YFINANCE_MIN_INTERVAL_SECONDS) - now
        if wait_for > 0:
            time.sleep(wait_for)
        _YFINANCE_LAST_CALL_AT = time.monotonic()


def _record_yfinance_result(params: dict[str, Any], exc: Exception | None) -> None:
    if not _uses_yfinance(params):
        return
    global _YFINANCE_COOLDOWN_UNTIL
    if exc is None:
        return
    if _is_openbb_rate_limit_error(exc):
        with _YFINANCE_CALL_LOCK:
            _YFINANCE_COOLDOWN_UNTIL = max(
                _YFINANCE_COOLDOWN_UNTIL,
                time.monotonic() + _YFINANCE_RATE_LIMIT_COOLDOWN_SECONDS,
            )


def _yfinance_cooldown_remaining(params: dict[str, Any]) -> int | None:
    if not _uses_yfinance(params):
        return None
    remaining = int(max(0.0, _YFINANCE_COOLDOWN_UNTIL - time.monotonic()))
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
