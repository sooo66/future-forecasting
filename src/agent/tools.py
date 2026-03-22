"""Minimal agent tools for search and historical market data."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import re
from typing import Any

from qwen_agent.tools.base import BaseTool
from qwen_agent.utils.tokenization_qwen import tokenizer as qwen_tokenizer

from tools.openbb import call_openbb_function, list_supported_openbb_functions
from tools.search import SearchClient, resolve_search_api_base, resolve_search_retrieval_mode
from tools.search_clients import build_search_client


DEFAULT_SEARCH_LIMIT = 3
DEFAULT_SEARCH_CONTENT_TOKENS = 256
DEFAULT_SEARCH_MAX_CALLS = 2
_DEFAULT_OPENBB_WINDOW_DAYS = 90
_ALLOWED_HISTORICAL_FUNCTIONS = {
    "equity.price.historical",
    "crypto.price.historical",
    "currency.price.historical",
    "index.price.historical",
}
_QUOTE_ALIASES = {
    "equity.price.quote": "equity.price.historical",
    "crypto.price.quote": "crypto.price.historical",
    "currency.price.quote": "currency.price.historical",
    "index.price.quote": "index.price.historical",
}


class SearchTool(BaseTool):
    name = "search"
    description = (
        "Search the local search API. The cutoff time is injected externally for the current question. "
        "Only provide the query and optional source filter."
    )
    parameters = [
        {"name": "question", "type": "string", "description": "Search query or keywords.", "required": True},
        {
            "name": "source",
            "type": "string",
            "description": "Optional source filter such as news, paper, blog, sociomedia, book, or report.",
            "required": False,
        },
    ]

    def __init__(
        self,
        *,
        search_client: SearchClient | Any,
        default_cuttime: str | None = None,
        limit: int = DEFAULT_SEARCH_LIMIT,
        content_tokens: int = DEFAULT_SEARCH_CONTENT_TOKENS,
        max_calls: int = DEFAULT_SEARCH_MAX_CALLS,
    ) -> None:
        super().__init__()
        self._search_client = search_client
        self.default_cuttime = (default_cuttime or "").strip() or None
        self.limit = max(1, int(limit or DEFAULT_SEARCH_LIMIT))
        self.content_tokens = max(1, int(content_tokens or DEFAULT_SEARCH_CONTENT_TOKENS))
        self.max_calls = max(1, int(max_calls or DEFAULT_SEARCH_MAX_CALLS))
        self._call_count = 0
        self._seen_queries: set[str] = set()

    def call(self, params: str | dict, **kwargs: Any) -> dict[str, Any]:
        payload = self._verify_json_format_args(params)
        question = str(payload.get("question") or "").strip()
        normalized_query = _normalize_search_query(question)
        if not normalized_query:
            return {"warning": "Empty search query.", "hits": []}
        if normalized_query in self._seen_queries:
            return {
                "warning": "Repeated search query blocked. Refine the query materially or finalize.",
                "hits": [],
            }
        if self._call_count >= self.max_calls:
            return {
                "warning": "Search budget exhausted. Use existing evidence, switch tools, or finalize.",
                "hits": [],
            }
        self._call_count += 1
        self._seen_queries.add(normalized_query)
        cuttime = _resolve_cuttime(default_cuttime=self.default_cuttime, runtime_kwargs=kwargs)
        result = self._search_client.search(
            question,
            time=cuttime,
            source=payload.get("source"),
            limit=self.limit,
        )
        return _truncate_search_payload(result, token_limit=self.content_tokens)

    def format_result_for_model(self, payload: dict[str, Any]) -> dict[str, Any]:
        hits = payload.get("hits")
        if not isinstance(hits, list):
            return payload
        compact_hits = [str(hit.get("content") or "").strip() for hit in hits if str(hit.get("content") or "").strip()]
        formatted: dict[str, Any] = {
            "count": len(compact_hits),
            "hits": compact_hits,
        }
        warning = str(payload.get("warning") or "").strip()
        if warning:
            formatted["warning"] = warning
        return formatted


class OpenBBTool(BaseTool):
    name = "openbb"
    description = (
        "Query historical-safe OpenBB functions. The cutoff time is injected externally for the "
        "current question and all dates will be clamped to it."
    )
    parameters = [
        {
            "name": "function",
            "type": "string",
            "description": (
                "Historical-safe OpenBB function name, such as equity.price.historical, index.price.historical, "
                "currency.price.historical, crypto.price.historical, or function='list'. "
                "Quote aliases are accepted and mapped to the latest historical value before the cutoff."
            ),
            "required": True,
        },
        {"name": "params", "type": "object", "description": "JSON params for the OpenBB function.", "required": False},
    ]

    def __init__(self, *, default_cuttime: str | None = None):
        super().__init__()
        self.default_cuttime = (default_cuttime or "").strip() or None

    def call(self, params: str | dict, **kwargs: Any) -> dict[str, Any]:
        payload = self._verify_json_format_args(params)
        raw_function = str(payload.get("function") or "").strip()
        if raw_function.lower() in {"", "list", "help"}:
            return {
                "supported_functions": self._supported_functions(),
                "notes": (
                    "Only historical-safe market data functions are enabled here. "
                    "All dates are clamped to the call-level cuttime."
                ),
                "default_cuttime": self.default_cuttime,
            }

        cuttime = _resolve_cuttime(default_cuttime=self.default_cuttime, runtime_kwargs=kwargs)
        cutoff_day = _normalize_date(cuttime)
        target_function, latest_only = self._normalize_function(raw_function)
        if target_function not in _ALLOWED_HISTORICAL_FUNCTIONS:
            return {
                "error": f"Unsupported or unsafe OpenBB function: {raw_function}",
                "supported_functions": self._supported_functions(),
                "cuttime": cuttime,
            }

        raw_params = payload.get("params")
        if raw_params is not None and not isinstance(raw_params, dict):
            return {"error": "openbb.params must be a JSON object", "cuttime": cuttime}

        prepared = self._prepare_params(dict(raw_params or {}), cutoff_day=cutoff_day)
        result = call_openbb_function(target_function, params=prepared, limit=None)
        result["cuttime"] = cuttime
        if latest_only and isinstance(result.get("results"), list) and result["results"]:
            result["results"] = [result["results"][-1]]
            result["result_count"] = 1
            result["requested_function"] = raw_function
            result["aliased_to"] = target_function
        return result

    @staticmethod
    def _prepare_params(params: dict[str, Any], *, cutoff_day: date) -> dict[str, Any]:
        end_date = _normalize_date(params.get("end_date")) if params.get("end_date") else cutoff_day
        if end_date > cutoff_day:
            end_date = cutoff_day
        if params.get("start_date"):
            start_date = _normalize_date(params.get("start_date"))
        else:
            start_date = end_date - timedelta(days=_DEFAULT_OPENBB_WINDOW_DAYS)
        if start_date > end_date:
            start_date = end_date - timedelta(days=_DEFAULT_OPENBB_WINDOW_DAYS)
        params["start_date"] = start_date.isoformat()
        params["end_date"] = end_date.isoformat()
        return params

    @staticmethod
    def _normalize_function(function: str) -> tuple[str, bool]:
        lowered = function.strip().lower()
        if lowered in _QUOTE_ALIASES:
            return _QUOTE_ALIASES[lowered], True
        return lowered, False

    def _supported_functions(self) -> list[dict[str, Any]]:
        allowed = []
        for item in list_supported_openbb_functions():
            if item["function"] in _ALLOWED_HISTORICAL_FUNCTIONS:
                allowed.append(item)
        allowed.extend(
            {
                "function": alias,
                "description": f"Alias for the latest historical observation from {target} before the cutoff.",
                "default_provider": "yfinance",
                "signature": "(symbol, optional start_date/end_date)",
                "example_params": {"symbol": "AAPL"},
            }
            for alias, target in sorted(_QUOTE_ALIASES.items())
        )
        return allowed


def build_default_tools(
    *,
    project_root: Path | str,
    search_api_base: str | None = None,
    search_retrieval_mode: str | None = None,
    search_backend: str | None = None,
    cutoff_time: str | None = None,
    search_limit: int = DEFAULT_SEARCH_LIMIT,
    enable_code_interpreter: bool = True,
) -> list[dict[str, Any] | BaseTool]:
    repo_root = Path(project_root).resolve()
    resolved_cutoff = (cutoff_time or "").strip() or _default_cutoff_time()
    tools: list[dict[str, Any] | BaseTool] = []
    if enable_code_interpreter:
        tools.append(
            {
                "name": "code_interpreter",
                "work_dir": str((repo_root / ".qwen_agent_workspace" / "code_interpreter").resolve()),
            }
        )
    search_client = build_search_client(
        base_url=resolve_search_api_base(search_api_base),
        default_mode=resolve_search_retrieval_mode(search_retrieval_mode),
        backend=search_backend,
    )
    tools.append(
        SearchTool(
            search_client=search_client,
            default_cuttime=resolved_cutoff,
            limit=search_limit,
        )
    )
    tools.append(OpenBBTool(default_cuttime=resolved_cutoff))
    return tools


def _default_cutoff_time() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _resolve_cuttime(*, default_cuttime: str | None, runtime_kwargs: dict[str, Any]) -> str:
    cuttime = str(
        runtime_kwargs.get("cuttime") or runtime_kwargs.get("cutoff_time") or default_cuttime or ""
    ).strip()
    if not cuttime:
        raise ValueError("cuttime is required for this tool call")
    return cuttime


def _truncate_search_payload(payload: dict[str, Any], *, token_limit: int) -> dict[str, Any]:
    hits = payload.get("hits")
    if not isinstance(hits, list):
        return payload
    copied = dict(payload)
    copied["hits"] = [_truncate_search_hit(hit, token_limit=token_limit) for hit in hits]
    return copied


def _truncate_search_hit(hit: dict[str, Any], *, token_limit: int) -> dict[str, Any]:
    copied = dict(hit)
    copied["content"] = _truncate_text_by_tokens(str(hit.get("content") or ""), token_limit=token_limit)
    return copied


def _truncate_text_by_tokens(text: str, *, token_limit: int) -> str:
    tokens = qwen_tokenizer.tokenize(text)
    if len(tokens) <= token_limit:
        return text
    return qwen_tokenizer.convert_tokens_to_string(tokens[:token_limit]).rstrip() + "..."


def _normalize_search_query(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^0-9a-zA-Z]+", " ", text.lower())).strip()


def _normalize_date(value: Any) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    text = str(value or "").strip()
    if not text:
        raise ValueError("date value is required")
    if "T" in text:
        text = text.split("T", 1)[0]
    return datetime.fromisoformat(text).date()
