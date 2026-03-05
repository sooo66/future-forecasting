"""ArXiv info module adapter into unified text schema."""
from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Iterable, Iterator, Optional
import xml.etree.ElementTree as ET

import requests
from loguru import logger

from core.contracts import TextRecord, stable_record_id
from modules.base import RunContext
from utils.time_utils import to_day, to_iso_utc


ARXIV_API = "https://export.arxiv.org/api/query"
PAGE_SIZE = 100


def _fmt_query_dt(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y%m%d%H%M")


def _clean(text: Optional[str]) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


class ArxivModule:
    name = "info.paper.arxiv"

    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.trust_env = False
        self.session.headers.update(
            {
                "Accept": "application/atom+xml",
                "User-Agent": "future-forecasting-arxiv-module/1.0",
            }
        )

    def run(self, ctx: RunContext) -> Iterable[dict]:
        logger.info(f"[{self.name}] importing arxiv from={ctx.date_from.date()} to={ctx.date_to.date()}")
        query = self._build_query(ctx.date_from, ctx.date_to)
        def _iter():
            count = 0
            for row in self._fetch(query):
                count += 1
                yield row
            logger.info(f"[{self.name}] normalized records={count}")

        return _iter()

    def _build_query(self, dt_from: datetime, dt_to: datetime) -> str:
        date_clause = f"submittedDate:[{_fmt_query_dt(dt_from)} TO {_fmt_query_dt(dt_to)}]"
        category_clause = " OR ".join(
            [
                "cat:cs.AI",
                "cat:cs.CL",
                "cat:cs.CY",
                "cat:cs.LG",
                "cat:stat.ML",
                "cat:q-fin.EC",
                "cat:econ.GN",
            ]
        )
        keyword_clause = " OR ".join(
            [
                "all:forecast",
                "all:prediction",
                'all:"time series"',
                "all:scenario",
                "all:risk",
                "all:climate",
                "all:economic",
            ]
        )
        return f"(({category_clause}) OR ({keyword_clause})) AND {date_clause}"

    def _fetch(self, query: str) -> Iterator[dict]:
        ns = {"atom": "http://www.w3.org/2005/Atom", "opensearch": "http://a9.com/-/spec/opensearch/1.1/"}
        start = 0
        total: Optional[int] = None

        while True:
            params = {
                "search_query": query,
                "start": start,
                "max_results": PAGE_SIZE,
                "sortBy": "submittedDate",
                "sortOrder": "ascending",
            }
            resp = self.session.get(ARXIV_API, params=params, timeout=(10, 60))
            if resp.status_code >= 400:
                break
            root = ET.fromstring(resp.text)
            if total is None:
                total_text = root.findtext("opensearch:totalResults", default="", namespaces=ns).strip()
                total = int(total_text) if total_text.isdigit() else None

            entries = root.findall("atom:entry", ns)
            if not entries:
                break

            for entry in entries:
                row = self._normalize_entry(entry, ns)
                if row:
                    yield row

            start += len(entries)
            if total is not None and start >= total:
                break

    def _normalize_entry(self, entry: ET.Element, ns: dict) -> Optional[dict]:
        title = _clean(entry.findtext("atom:title", default="", namespaces=ns))
        summary = _clean(entry.findtext("atom:summary", default="", namespaces=ns))
        published = _clean(entry.findtext("atom:published", default="", namespaces=ns))
        url = self._entry_url(entry, ns)
        if not title or not url:
            return None

        ts = to_iso_utc(published) or published
        day = to_day(ts)
        if not day:
            return None

        authors = []
        for author in entry.findall("atom:author", ns):
            name = _clean(author.findtext("atom:name", default="", namespaces=ns))
            if name:
                authors.append(name)
        categories = []
        for cat in entry.findall("atom:category", ns):
            term = cat.get("term")
            if term:
                categories.append(term)

        source = "paper/arxiv"
        payload = {
            "title": title,
            "description": summary or None,
            "content": summary,
            "authors": authors,
            "categories": categories,
        }
        rid = stable_record_id(source, url, day, title)
        return TextRecord(
            id=rid,
            kind="info",
            source=source,
            timestamp=day,
            url=url,
            payload=payload,
        ).normalized().to_dict()

    @staticmethod
    def _entry_url(entry: ET.Element, ns: dict) -> str:
        for link in entry.findall("atom:link", ns):
            if link.get("rel") == "alternate":
                href = link.get("href")
                if href:
                    return href
        return _clean(entry.findtext("atom:id", default="", namespaces=ns))
