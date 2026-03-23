"""Exa-backed search client compatible with the local search interface."""

from __future__ import annotations

import os
from typing import Any, Sequence

from utils.env import get_first_env

_DEFAULT_EXA_BASE_URL = "https://api.exa.ai"
_DEFAULT_EXA_SEARCH_TYPE = "auto"
_DEFAULT_TEXT_MAX_CHARACTERS = 4000
_EXA_CATEGORY_BY_SOURCE = {
    "news": "news",
    "paper": "research paper",
    "report": "financial report",
    "sociomedia": "tweet",
}


class ExaSearchClient:
    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        search_type: str = _DEFAULT_EXA_SEARCH_TYPE,
        text_max_characters: int = _DEFAULT_TEXT_MAX_CHARACTERS,
    ) -> None:
        resolved_api_key = str(
            api_key or get_first_env("EXA_API_KEY", "EXA_SEARCH_API_KEY")
        ).strip()
        if not resolved_api_key:
            raise ValueError(
                "EXA_API_KEY (or EXA_SEARCH_API_KEY) is required for Exa search"
            )
        self.api_key = resolved_api_key
        self.base_url = str(
            base_url or get_first_env("EXA_BASE_URL") or _DEFAULT_EXA_BASE_URL
        ).rstrip("/")
        self.search_type = (
            str(search_type or _DEFAULT_EXA_SEARCH_TYPE).strip()
            or _DEFAULT_EXA_SEARCH_TYPE
        )
        self.text_max_characters = max(
            128, int(text_max_characters or _DEFAULT_TEXT_MAX_CHARACTERS)
        )
        self._client = _load_exa_class()(self.api_key, base_url=self.base_url)

    def search(
        self,
        question: str,
        *,
        time: str | None = None,
        source: str | Sequence[str] | None = None,
        limit: int = 3,
        mode: str | None = None,
    ) -> dict[str, Any]:
        query = str(question or "").strip()
        if not query:
            raise ValueError("search.question is required")
        source_type = _normalize_source_type(source)
        category = _EXA_CATEGORY_BY_SOURCE.get(source_type or "")
        try:
            response = self._client.search(
                query,
                end_published_date=time,
                num_results=max(1, int(limit or 5)),
                type=self.search_type,
                category=category,
                contents={
                    "text": {
                        "max_characters": self.text_max_characters,
                        "verbosity": "compact",
                    }
                },
            )
        except Exception as exc:
            proxy_hint = _active_proxy_summary()
            message = f"Exa search request failed: {exc}"
            if proxy_hint:
                message += f". Active proxy env: {proxy_hint}"
            raise RuntimeError(message) from exc
        hits = []
        for result in list(getattr(response, "results", []) or []):
            hits.append(_exa_result_to_hit(result, requested_source=source_type))
        return {
            "snapshot_root": "exa",
            "backend": "exa",
            "question": query,
            "time": time,
            "source": source_type or "",
            "mode": "exa",
            "total_candidates": len(hits),
            "hits": hits,
        }

    def health(self) -> dict[str, Any]:
        return {
            "backend": "exa",
            "snapshot_root": "exa",
            "search_root": None,
            "default_mode": "exa",
            "available_modes": ["exa"],
            "base_url": self.base_url,
        }


def _load_exa_class() -> Any:
    try:
        from exa_py import Exa
    except ImportError as exc:
        raise RuntimeError(
            "exa-py is required for Exa search. Install project dependencies first."
        ) from exc
    return Exa


def _normalize_source_type(source: str | Sequence[str] | None) -> str:
    if source is None:
        return ""
    if isinstance(source, str):
        candidates = [
            item.strip().lower() for item in source.split(",") if item.strip()
        ]
    else:
        candidates = [str(item).strip().lower() for item in source if str(item).strip()]
    for item in candidates:
        if "/" in item:
            return item.split("/", 1)[-1]
        return item
    return ""


def _exa_result_to_hit(result: Any, *, requested_source: str) -> dict[str, Any]:
    title = str(_result_field(result, "title", "") or "").strip()
    content = _compose_content(title, _extract_result_text(result))
    url = str(_result_field(result, "url", "") or "").strip() or None
    published_date = _extract_published_date(result)
    score = _result_field(result, "score", None)
    source_type = requested_source or _infer_source_type(result)
    return {
        "doc_id": str(_result_field(result, "id", "") or url or title),
        "score": round(float(score or 0.0), 4),
        "source": f"info/{source_type}" if source_type else "info/news",
        "source_type": source_type or "news",
        "timestamp": published_date[:10] if published_date else "",
        "title": title,
        "content": content,
        "url": url,
    }


def _extract_result_text(result: Any) -> str:
    summary = str(_result_field(result, "summary", "") or "").strip()
    if summary:
        return summary
    highlights = _result_field(result, "highlights", None)
    if isinstance(highlights, list):
        joined = " ".join(str(item).strip() for item in highlights if str(item).strip())
        if joined:
            return joined
    text = str(_result_field(result, "text", "") or "").strip()
    return _clean_result_text(text)


def _infer_source_type(result: Any) -> str:
    url = str(_result_field(result, "url", "") or "").lower()
    if url.endswith(".pdf"):
        return "paper"
    return "news"


def _extract_published_date(result: Any) -> str:
    direct = str(_result_field(result, "published_date", "") or "").strip()
    if direct:
        return direct
    extras = _result_field(result, "extras", None)
    if isinstance(extras, dict):
        for key in ("published_date", "publishedDate", "published_at", "publishedAt"):
            value = str(extras.get(key) or "").strip()
            if value:
                return value
    return ""


def _compose_content(title: str, content: str) -> str:
    title = title.strip()
    content = content.strip()
    if not title:
        return content
    lowered_prefix = content[: len(title) + 8].lower()
    if content and title.lower() in lowered_prefix:
        return content
    if content:
        return f"{title}\n{content}"
    return title


def _clean_result_text(text: str) -> str:
    if not text:
        return ""
    cleaned = "\n".join(line.strip() for line in str(text).splitlines())
    cleaned = "\n".join(line for line in cleaned.splitlines() if line)
    return cleaned.strip()


def _result_field(result: Any, name: str, default: Any = None) -> Any:
    if isinstance(result, dict):
        return result.get(name, default)
    return getattr(result, name, default)


def _active_proxy_summary() -> str:
    entries = []
    for key in (
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
    ):
        value = os.getenv(key, "").strip()
        if value:
            entries.append(f"{key}={value}")
    return ", ".join(entries)
