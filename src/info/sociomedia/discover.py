# -*- coding: utf-8 -*-
"""URL discovery for sociomedia sources."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from bs4 import BeautifulSoup
from loguru import logger

from utils.importer_base import BaseImporter


DEFAULT_QUORA_KEYWORDS = [
    "artificial intelligence",
    "machine learning",
    "robotics",
    "economy",
    "inflation",
    "geopolitics",
    "climate",
    "energy",
    "cybersecurity",
    "startup",
    "biotech",
    "space",
    "semiconductor",
    "quantum",
    "healthcare",
]

FIXED_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
)


@dataclass(frozen=True)
class QuoraDiscoverConfig:
    per_day_limit: int = 200
    keywords: List[str] | None = None


class QuoraDiscoverer(BaseImporter):
    def __init__(self, config: QuoraDiscoverConfig) -> None:
        super().__init__(timeout=20, retries=2, backoff=1.0, user_agent=FIXED_USER_AGENT)
        self.config = config

    def discover_urls(self) -> List[str]:
        keywords = self.config.keywords or DEFAULT_QUORA_KEYWORDS
        urls: List[str] = []
        seen = set()

        for keyword in keywords:
            query = keyword.replace(" ", "+")
            search_url = f"https://www.quora.com/search?q={query}"
            try:
                html = self._request_text(search_url, accept="text/html")
            except RuntimeError as exc:
                logger.warning(f"Quora search failed: {search_url} ({exc})")
                continue
            soup = BeautifulSoup(html, "lxml")
            for link in soup.find_all("a", href=True):
                href = str(link.get("href") or "")
                if "/answer/" not in href and "/answers/" not in href:
                    continue
                if href.startswith("/"):
                    href = f"https://www.quora.com{href}"
                if href in seen:
                    continue
                seen.add(href)
                urls.append(href)
                if len(urls) >= self.config.per_day_limit:
                    return urls

            if len(urls) >= self.config.per_day_limit:
                break

        logger.info(f"Discovered {len(urls)} Quora URLs")
        return urls

    def run(self) -> List[str]:
        return self.discover_urls()
