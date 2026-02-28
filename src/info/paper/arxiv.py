"""ArXiv API crawler for paper records."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
from uuid import uuid4
import xml.etree.ElementTree as ET

from loguru import logger

from utils.models import Record
from utils.importer_base import BaseImporter


ARXIV_API = "http://export.arxiv.org/api/query"
DEFAULT_PAGE_SIZE = 100
DEFAULT_TIMEOUT = 20


@dataclass(frozen=True)
class ArxivConfig:
    start_date: str
    end_date: str
    output_path: Path
    page_size: int = DEFAULT_PAGE_SIZE
    timeout: int = DEFAULT_TIMEOUT
    retries: int = 0
    backoff: float = 0.0
    user_agent: str = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    )


class ArxivImportError(RuntimeError):
    """Raised when ArXiv import fails in a non-recoverable way."""


class ArxivImporter(BaseImporter):
    def __init__(self, config: ArxivConfig) -> None:
        super().__init__(
            timeout=config.timeout,
            retries=config.retries,
            backoff=config.backoff,
            user_agent=config.user_agent,
        )
        self.config = config

    def run(self) -> List[Record]:
        start_dt = self._parse_datetime(self.config.start_date)
        end_dt = self._parse_datetime(self.config.end_date)
        if not start_dt or not end_dt:
            raise ArxivImportError("Invalid start_date or end_date")
        if start_dt > end_dt:
            raise ArxivImportError("start_date must be <= end_date")

        logger.info(
            f"ArXiv fetch start={start_dt.isoformat()} end={end_dt.isoformat()} "
            f"page_size={self.config.page_size}"
        )

        query = _build_query(start_dt, end_dt)
        total = None
        start = 0
        records: List[Record] = []

        while True:
            params = {
                "search_query": query,
                "start": start,
                "max_results": self.config.page_size,
                "sortBy": "submittedDate",
                "sortOrder": "ascending",
            }
            try:
                payload = self._request_text(ARXIV_API, params=params, accept="application/atom+xml")
            except RuntimeError as exc:
                raise ArxivImportError(str(exc)) from exc
            feed = _parse_feed(payload)
            if total is None:
                total = feed.total_results
                logger.info(f"ArXiv total_results={total}")
            if not feed.entries:
                break

            for entry in feed.entries:
                rec = _record_from_entry(entry)
                if rec:
                    records.append(rec)

            start += len(feed.entries)
            logger.info(f"Fetched {len(records)} records (start={start})")
            if total is not None and start >= total:
                break

        return records


def _build_query(start_dt: datetime, end_dt: datetime) -> str:
    start_str = start_dt.strftime("%Y%m%d%H%M")
    end_str = end_dt.strftime("%Y%m%d%H%M")
    return f"all:* AND submittedDate:[{start_str} TO {end_str}]"


@dataclass(frozen=True)
class _FeedEntry:
    entry_id: str
    url: str
    title: str
    summary: str
    published: str
    updated: str
    categories: List[str]


@dataclass(frozen=True)
class _Feed:
    total_results: Optional[int]
    entries: List[_FeedEntry]


def _parse_feed(xml_text: str) -> _Feed:
    root = ET.fromstring(xml_text)
    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
    }

    total_results = _get_text(root, "opensearch:totalResults", ns)
    total = int(total_results) if total_results and total_results.isdigit() else None

    entries: List[_FeedEntry] = []
    for entry in root.findall("atom:entry", ns):
        entry_id = _get_text(entry, "atom:id", ns) or ""
        title = _clean_text(_get_text(entry, "atom:title", ns))
        summary = _clean_text(_get_text(entry, "atom:summary", ns))
        published = _get_text(entry, "atom:published", ns) or ""
        updated = _get_text(entry, "atom:updated", ns) or ""
        url = _get_entry_link(entry, ns)
        categories: List[str] = []
        for cat in entry.findall("atom:category", ns):
            term = cat.get("term")
            if term:
                categories.append(term)

        entries.append(
            _FeedEntry(
                entry_id=entry_id,
                url=url,
                title=title,
                summary=summary,
                published=published,
                updated=updated,
                categories=categories,
            )
        )

    return _Feed(total_results=total, entries=entries)


def _get_text(node: ET.Element, path: str, ns: dict) -> Optional[str]:
    target = node.find(path, ns)
    if target is None or target.text is None:
        return None
    return target.text.strip()


def _get_entry_link(entry: ET.Element, ns: dict) -> str:
    for link in entry.findall("atom:link", ns):
        if link.get("rel") == "alternate":
            href = link.get("href")
            if href:
                return href
    return entry.get("href") or ""


def _clean_text(text: Optional[str]) -> str:
    if not text:
        return ""
    return " ".join(text.split())


def _record_from_entry(entry: _FeedEntry) -> Optional[Record]:
    if not entry.url or not entry.title:
        return None
    published_dt = BaseImporter._parse_datetime(entry.published) or BaseImporter._parse_datetime(entry.updated)
    if not published_dt:
        return None
    pubtime = published_dt.isoformat()
    return Record(
        id=str(uuid4()),
        source="arxiv",
        url=entry.url,
        title=entry.title,
        description=entry.summary,
        content=entry.summary,
        published_at=pubtime,
        language="en",
        tags=entry.categories or None,
        pubtime=pubtime,
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="ArXiv API crawler")
    parser.add_argument("--start-date", required=True, help="ISO8601 start datetime")
    parser.add_argument("--end-date", required=True, help="ISO8601 end datetime")
    parser.add_argument(
        "--output",
        default="data/processed/arxiv/arxiv.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    args = parser.parse_args()

    config = ArxivConfig(
        start_date=args.start_date,
        end_date=args.end_date,
        output_path=Path(args.output),
        page_size=args.page_size,
        timeout=args.timeout,
    )
    importer = ArxivImporter(config)
    records = importer.run()
    importer._write_jsonl(records, config.output_path)
    logger.info(f"Wrote {len(records)} records to {config.output_path}")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    main()
