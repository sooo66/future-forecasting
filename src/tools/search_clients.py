"""Search client factory helpers."""

from __future__ import annotations

import os
from typing import Any

from tools.exa_search import ExaSearchClient
from tools.search import SearchClient


def resolve_search_backend(backend: str | None = None) -> str:
    value = str(backend or os.getenv("SEARCH_BACKEND") or "local").strip().lower()
    if value not in {"local", "exa"}:
        raise ValueError(f"Unsupported search backend: {value}")
    return value


def build_search_client(
    *,
    base_url: str | None = None,
    default_mode: str | None = None,
    backend: str | None = None,
    exa_base_url: str | None = None,
) -> Any:
    resolved_backend = resolve_search_backend(backend)
    if resolved_backend == "exa":
        return ExaSearchClient(base_url=exa_base_url)
    return SearchClient(base_url, default_mode=default_mode)
