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
import time
from typing import List, Optional
from uuid import uuid4
import xml.etree.ElementTree as ET

from loguru import logger
import requests

from utils.importer_base import BaseImporter
from utils.models import Record


ARXIV_API = "https://export.arxiv.org/api/query"
DEFAULT_PAGE_SIZE = 100
DEFAULT_TIMEOUT = 20
DEFAULT_QUERY_PROFILE = "forecasting"
DEFAULT_OUTPUT_PATH = "data/processed/arxiv/arxiv.jsonl"

# Broad and forecasting-friendly domains spanning CS/Stats/Econ/Finance/Society/Earth/Bio.
FORECASTING_CATEGORIES = (
    "cs.AI",
    "cs.CL",
    "cs.CY",
    "cs.LG",
    "cs.SI",
    "econ.EM",
    "econ.GN",
    "q-fin.EC",
    "q-fin.RM",
    "q-fin.ST",
    "stat.AP",
    "stat.ML",
    "physics.geo-ph",
    "physics.soc-ph",
    "q-bio.PE",
    "q-bio.QM",
)


@dataclass(frozen=True)
class ArxivConfig:
    from_date: str
    to_date: str
    output_path: Path
    page_size: int = DEFAULT_PAGE_SIZE
    timeout: int = DEFAULT_TIMEOUT
    query_profile: str = DEFAULT_QUERY_PROFILE
    custom_query: Optional[str] = None
    retries: int = 3
    backoff: float = 1.0
    request_interval: float = 3.0
    max_429_retries: int = 6
    use_env_proxy: bool = False
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
        self.session.trust_env = config.use_env_proxy

    def run(self) -> List[Record]:
        from_dt = self._parse_datetime(self.config.from_date)
        to_dt = self._parse_datetime(self.config.to_date)
        if not from_dt or not to_dt:
            raise ArxivImportError("Invalid from_date or to_date")
        if from_dt > to_dt:
            raise ArxivImportError("from_date must be <= to_date")

        logger.info(
            f"ArXiv fetch from={from_dt.isoformat()} to={to_dt.isoformat()} "
            f"page_size={self.config.page_size} profile={self.config.query_profile}"
        )

        query = _build_query(
            from_dt,
            to_dt,
            query_profile=self.config.query_profile,
            custom_query=self.config.custom_query,
        )
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
                payload = self._request_feed(params=params)
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
            if self.config.request_interval > 0:
                time.sleep(self.config.request_interval)

        return records

    def _request_feed(self, *, params: dict) -> str:
        headers = {"Accept": "application/atom+xml", "User-Agent": self.user_agent}
        for attempt in range(self.config.max_429_retries + 1):
            try:
                response = self.session.get(
                    ARXIV_API,
                    params=params,
                    headers=headers,
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                logger.error(f"Request failed: {ARXIV_API} ({exc})")
                raise RuntimeError(f"Request failed: {ARXIV_API}") from exc

            if response.status_code == 429:
                if attempt >= self.config.max_429_retries:
                    raise RuntimeError(f"Request failed: {ARXIV_API} (429 rate limited)")
                wait_s = max(self.config.request_interval, 1.0) * (2 ** attempt)
                logger.warning(
                    f"ArXiv rate limited (429), retry in {wait_s:.1f}s "
                    f"(attempt {attempt + 1}/{self.config.max_429_retries})"
                )
                time.sleep(wait_s)
                continue

            try:
                response.raise_for_status()
            except requests.RequestException as exc:
                logger.error(f"Request failed: {ARXIV_API} ({exc})")
                raise RuntimeError(f"Request failed: {ARXIV_API}") from exc
            return response.text

        raise RuntimeError(f"Request failed: {ARXIV_API}")


def _normalize_date(value: str, *, is_end: bool) -> str:
    if "T" in value or ":" in value:
        return value
    if is_end:
        return f"{value}T23:59:59+00:00"
    return f"{value}T00:00:00+00:00"


def _build_query(
    from_dt: datetime,
    to_dt: datetime,
    *,
    query_profile: str,
    custom_query: Optional[str],
) -> str:
    from_str = from_dt.strftime("%Y%m%d%H%M")
    to_str = to_dt.strftime("%Y%m%d%H%M")
    date_clause = f"submittedDate:[{from_str} TO {to_str}]"

    if custom_query:
        domain_clause = custom_query.strip()
    elif query_profile == "all":
        domain_clause = "all:*"
    elif query_profile == "forecasting":
        category_clause = " OR ".join([f"cat:{cat}" for cat in FORECASTING_CATEGORIES])
        keyword_clause = " OR ".join(
            [
                "all:forecast",
                "all:prediction",
                'all:"time series"',
                "all:scenario",
                "all:risk",
                "all:policy",
                "all:climate",
                "all:economic",
                "all:epidemic",
                "all:energy",
            ]
        )
        domain_clause = f"(({category_clause}) OR ({keyword_clause}))"
    else:
        raise ArxivImportError(f"Unsupported query_profile: {query_profile}")

    return f"({domain_clause}) AND {date_clause}"


@dataclass(frozen=True)
class _FeedEntry:
    entry_id: str
    url: str
    pdf_url: str
    title: str
    summary: str
    published: str
    updated: str
    authors: List[str]
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
        pdf_url = _get_pdf_link(entry, ns)
        authors = _get_authors(entry, ns)
        categories: List[str] = []
        for cat in entry.findall("atom:category", ns):
            term = cat.get("term")
            if term:
                categories.append(term)

        entries.append(
            _FeedEntry(
                entry_id=entry_id,
                url=url,
                pdf_url=pdf_url,
                title=title,
                summary=summary,
                published=published,
                updated=updated,
                authors=authors,
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


def _get_pdf_link(entry: ET.Element, ns: dict) -> str:
    for link in entry.findall("atom:link", ns):
        href = link.get("href")
        title = (link.get("title") or "").lower()
        link_type = (link.get("type") or "").lower()
        if href and ("pdf" in title or "pdf" in link_type):
            return href
    return ""


def _get_authors(entry: ET.Element, ns: dict) -> List[str]:
    authors: List[str] = []
    for author in entry.findall("atom:author", ns):
        name = _get_text(author, "atom:name", ns)
        if name:
            authors.append(name)
    return authors


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
    payload = {
        "summary": entry.summary,
        "authors": entry.authors,
        "categories": entry.categories,
        "arxiv_id": entry.entry_id,
        "pdf_url": entry.pdf_url or None,
        "updated_at": entry.updated or None,
    }
    return Record(
        id=str(uuid4()),
        source="arxiv",
        url=entry.url,
        title=entry.title,
        description=entry.summary,
        content=payload,
        published_at=pubtime,
        language="en",
        tags=entry.categories or None,
        pubtime=pubtime,
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="ArXiv API crawler")
    parser.add_argument("--from", dest="from_date", required=True, help="YYYY-MM-DD or ISO8601")
    parser.add_argument("--to", dest="to_date", required=True, help="YYYY-MM-DD or ISO8601")
    parser.add_argument("--start-date", dest="start_date", help="Deprecated alias of --from")
    parser.add_argument("--end-date", dest="end_date", help="Deprecated alias of --to")
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help="Output JSONL path",
    )
    parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument(
        "--request-interval",
        type=float,
        default=3.0,
        help="Seconds between page requests to avoid arXiv rate limit (default: 3.0)",
    )
    parser.add_argument(
        "--max-429-retries",
        type=int,
        default=6,
        help="Max retries when arXiv returns HTTP 429 (default: 6)",
    )
    parser.add_argument(
        "--use-env-proxy",
        action="store_true",
        help="Use proxy settings from environment variables",
    )
    parser.add_argument(
        "--query-profile",
        choices=["forecasting", "all"],
        default=DEFAULT_QUERY_PROFILE,
        help="Preset query profile: forecasting (default) or all",
    )
    parser.add_argument(
        "--query",
        dest="custom_query",
        help="Custom arXiv search query (without submittedDate clause)",
    )
    args = parser.parse_args()

    from_raw = args.start_date or args.from_date
    to_raw = args.end_date or args.to_date
    from_date = _normalize_date(from_raw, is_end=False)
    to_date = _normalize_date(to_raw, is_end=True)

    config = ArxivConfig(
        from_date=from_date,
        to_date=to_date,
        output_path=Path(args.output),
        page_size=args.page_size,
        timeout=args.timeout,
        query_profile=args.query_profile,
        custom_query=args.custom_query,
        request_interval=args.request_interval,
        max_429_retries=args.max_429_retries,
        use_env_proxy=args.use_env_proxy,
    )
    importer = ArxivImporter(config)
    records = importer.run()
    importer._write_jsonl(records, config.output_path)
    logger.info(f"Wrote {len(records)} records to {config.output_path}")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    main()
