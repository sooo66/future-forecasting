#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import os
import re
import tempfile
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from PyPDF2 import PdfReader

BOOKS_API_URL = "https://openstax.org/apps/cms/api/books/?format=json"
SITEMAP_URL = "https://openstax.org/sitemap.xml"
SOURCE = "kb/openstax"


@dataclass
class HarvestPaths:
    root: Path
    text_dir: Path
    metadata_dir: Path
    state_dir: Path
    records_path: Path
    manifest_path: Path
    stats_path: Path
    progress_path: Path


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2))


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def ensure_dirs(out_root: Path) -> HarvestPaths:
    text_dir = out_root / "text"
    metadata_dir = out_root / "metadata"
    state_dir = out_root / "state"
    text_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    return HarvestPaths(
        root=out_root,
        text_dir=text_dir,
        metadata_dir=metadata_dir,
        state_dir=state_dir,
        records_path=metadata_dir / "openstax_records.jsonl",
        manifest_path=metadata_dir / "manifest.jsonl",
        stats_path=metadata_dir / "stats.json",
        progress_path=state_dir / "progress.json",
    )


def load_progress(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
            if isinstance(payload, dict):
                return payload
            return {}
    except Exception:
        return {}


def save_progress(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_json(path, payload)


def parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    for candidate in [text, text.replace("Z", "+00:00")]:
        try:
            dt = datetime.fromisoformat(candidate)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except ValueError:
            pass
    for fmt in ("%Y-%m-%d", "%b %d, %Y"):
        try:
            dt = datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            pass
    return None


def parse_cutoff(value: str) -> datetime:
    dt = parse_iso_datetime(value)
    if not dt:
        raise ValueError(f"Invalid cutoff datetime: {value}")
    return dt


class HTTPClient:
    def __init__(self, rate: float, max_retries: int = 4):
        self.session = requests.Session()
        self.rate = max(rate, 0.001)
        self.min_interval = 1.0 / self.rate
        self.max_retries = max_retries
        self._last_ts = 0.0

    def close(self) -> None:
        self.session.close()

    def _throttle(self) -> None:
        now = time.monotonic()
        wait = self.min_interval - (now - self._last_ts)
        if wait > 0:
            time.sleep(wait)
        self._last_ts = time.monotonic()

    def get(self, url: str, stream: bool = False) -> requests.Response:
        retryable_codes = {429, 500, 502, 503, 504}
        backoff = 1.0
        last_exc: Exception | None = None
        headers = {
            "Accept": "application/json, application/xml;q=0.9, text/html;q=0.8, */*;q=0.5",
            "User-Agent": "future-forecasting-openstax-harvester/1.0",
        }

        for attempt in range(1, self.max_retries + 1):
            self._throttle()
            try:
                resp = self.session.get(url, timeout=(10, 120), allow_redirects=True, stream=stream, headers=headers)
                if resp.status_code in retryable_codes and attempt < self.max_retries:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                return resp
            except requests.RequestException as exc:
                last_exc = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(backoff)
                backoff *= 2
        if last_exc:
            raise last_exc
        raise RuntimeError("HTTP request failed without explicit exception")


def get_books_catalog(client: HTTPClient) -> list[dict[str, Any]]:
    resp = client.get(BOOKS_API_URL)
    resp.encoding = "utf-8"
    if resp.status_code >= 400:
        raise RuntimeError(f"Books API error: HTTP {resp.status_code}")
    payload = resp.json()
    if not isinstance(payload, dict):
        raise RuntimeError("Books API payload is not a JSON object")
    books = payload.get("books")
    if not isinstance(books, list):
        raise RuntimeError("Books API payload has no books[]")
    return [b for b in books if isinstance(b, dict)]


def get_details_lastmod_map(client: HTTPClient) -> dict[str, str]:
    resp = client.get(SITEMAP_URL)
    resp.encoding = "utf-8"
    if resp.status_code >= 400:
        raise RuntimeError(f"Sitemap error: HTTP {resp.status_code}")

    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    root = ET.fromstring(resp.text)
    mapping: dict[str, str] = {}
    for url_el in root.findall("sm:url", ns):
        loc = url_el.findtext("sm:loc", default="", namespaces=ns).strip()
        lastmod = url_el.findtext("sm:lastmod", default="", namespaces=ns).strip()
        if "/details/books/" not in loc:
            continue
        if not lastmod:
            continue
        slug = loc.rstrip("/").split("/details/books/")[-1]
        if slug:
            mapping[slug] = lastmod
    return mapping


def normalize_slug(slug: str) -> str:
    text = slug.strip("/")
    if text.startswith("books/"):
        text = text.split("books/", 1)[1]
    return text


def choose_pubtime(book: dict[str, Any], details_lastmod: str | None) -> tuple[datetime | None, list[tuple[str, str]]]:
    evidence: list[tuple[str, str]] = []
    candidates: list[datetime] = []

    last_updated_pdf = book.get("last_updated_pdf")
    if isinstance(last_updated_pdf, str) and last_updated_pdf.strip():
        dt = parse_iso_datetime(last_updated_pdf)
        if dt:
            candidates.append(dt)
            evidence.append(("last_updated_pdf", last_updated_pdf))

    if details_lastmod:
        dt = parse_iso_datetime(details_lastmod)
        if dt:
            candidates.append(dt)
            evidence.append(("details_lastmod", details_lastmod))

    if not candidates:
        return None, evidence
    # Conservative gate: if multiple timestamps exist, use the latest one.
    return max(candidates), evidence


def safe_filename(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._")
    return cleaned or "unknown"


def primary_subject(subjects: list[str]) -> str:
    if subjects:
        return subjects[0]
    return "Unknown"


def parse_csv_list(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def choose_balanced_candidates(
    candidates: list[dict[str, Any]],
    target_downloads: int,
    max_per_subject: int,
) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in candidates:
        buckets[item["primary_subject"]].append(item)

    for subject in buckets:
        buckets[subject].sort(key=lambda x: (x["pubtime"], x["title"], x["doc_id"]))

    selected: list[dict[str, Any]] = []
    per_subject_count: Counter[str] = Counter()
    subjects_order = sorted(buckets.keys())

    while True:
        progressed = False
        for subject in subjects_order:
            group = buckets.get(subject, [])
            if not group:
                continue
            if max_per_subject > 0 and per_subject_count[subject] >= max_per_subject:
                continue
            selected.append(group.pop(0))
            per_subject_count[subject] += 1
            progressed = True
            if target_downloads > 0 and len(selected) >= target_downloads:
                return selected
        if not progressed:
            break

    return selected


def count_tokens(text: str) -> int:
    return len(re.findall(r"\w+", text, flags=re.UNICODE))


def extract_pdf_text(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    chunks: list[str] = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        page_text = page_text.replace("\x00", "")
        if page_text.strip():
            chunks.append(page_text.strip())
    return "\n\n".join(chunks).strip()


def log(msg: str) -> None:
    print(msg, flush=True)


def harvest(args: argparse.Namespace) -> int:
    cutoff_dt = parse_cutoff(args.cutoff)
    out_root = Path(args.out).resolve()
    paths = ensure_dirs(out_root)
    client = HTTPClient(rate=args.rate)

    counters: Counter[str] = Counter()
    runtime_stats = {
        "file_count": 0,
        "total_tokens": 0,
        "subject_distribution": Counter(),
        "subject_distribution_primary": Counter(),
        "pubtime_distribution": Counter(),
    }

    progress = load_progress(paths.progress_path) if args.resume else {}
    seen_ids = set(progress.get("done_ids", [])) if isinstance(progress.get("done_ids"), list) else set()

    try:
        books = get_books_catalog(client)
        details_lastmod_map = get_details_lastmod_map(client)
        counters["books_total"] = len(books)

        target_downloads = args.target_downloads if args.target_downloads > 0 else args.limit
        subject_whitelist = set(parse_csv_list(args.subject_whitelist)) if args.subject_whitelist else set()

        log(
            f"[START] cutoff={cutoff_dt.isoformat()} out={paths.root} books_total={len(books)} "
            f"target_downloads={target_downloads or 'all'} balanced={args.balance_subjects} "
            f"max_per_subject={args.max_per_subject or 'none'} resume={args.resume}"
        )

        eligible: list[dict[str, Any]] = []
        for book in books:

            state = str(book.get("book_state", "")).strip().lower()
            if state != "live":
                counters["skipped_not_live"] += 1
                continue

            slug_raw = str(book.get("slug", "")).strip()
            slug = normalize_slug(slug_raw)
            if not slug:
                counters["skipped_missing_slug"] += 1
                continue

            doc_id = f"openstax_{slug.replace('/', '_')}"
            if doc_id in seen_ids:
                counters["skipped_already_done"] += 1
                continue

            details_lastmod = details_lastmod_map.get(slug)
            pubtime_dt, evidence = choose_pubtime(book, details_lastmod)

            if pubtime_dt is None:
                counters["skipped_no_pubtime"] += 1
                append_jsonl(
                    paths.manifest_path,
                    {
                        "doc_id": doc_id,
                        "title": book.get("title", ""),
                        "status": "skipped_no_pubtime",
                        "evidence": [{"field": k, "value": v} for k, v in evidence],
                        "updated_at": now_iso(),
                    },
                )
                continue

            if pubtime_dt > cutoff_dt:
                counters["skipped_after_cutoff"] += 1
                append_jsonl(
                    paths.manifest_path,
                    {
                        "doc_id": doc_id,
                        "title": book.get("title", ""),
                        "status": "skipped_after_cutoff",
                        "pubtime": pubtime_dt.isoformat(),
                        "cutoff": cutoff_dt.isoformat(),
                        "evidence": [{"field": k, "value": v} for k, v in evidence],
                        "updated_at": now_iso(),
                    },
                )
                continue

            pdf_url = str(book.get("high_resolution_pdf_url") or book.get("low_resolution_pdf_url") or "").strip()
            if not pdf_url:
                counters["skipped_missing_pdf_url"] += 1
                append_jsonl(
                    paths.manifest_path,
                    {
                        "doc_id": doc_id,
                        "title": book.get("title", ""),
                        "status": "skipped_missing_pdf_url",
                        "pubtime": pubtime_dt.isoformat(),
                        "evidence": [{"field": k, "value": v} for k, v in evidence],
                        "updated_at": now_iso(),
                    },
                )
                continue

            subjects = book.get("subjects") if isinstance(book.get("subjects"), list) else []
            subjects = [str(x) for x in subjects if isinstance(x, str) and x.strip()]
            book_primary_subject = primary_subject(subjects)
            if subject_whitelist and book_primary_subject not in subject_whitelist:
                counters["skipped_subject_not_selected"] += 1
                append_jsonl(
                    paths.manifest_path,
                    {
                        "doc_id": doc_id,
                        "title": book.get("title", ""),
                        "status": "skipped_subject_not_selected",
                        "primary_subject": book_primary_subject,
                        "subjects": subjects,
                        "pubtime": pubtime_dt.isoformat(),
                        "evidence": [{"field": k, "value": v} for k, v in evidence],
                        "updated_at": now_iso(),
                    },
                )
                continue

            eligible.append(
                {
                    "doc_id": doc_id,
                    "title": str(book.get("title", "")),
                    "slug": slug,
                    "book_state": state,
                    "book": book,
                    "pubtime": pubtime_dt,
                    "evidence": evidence,
                    "pdf_url": pdf_url,
                    "subjects": subjects,
                    "primary_subject": book_primary_subject,
                }
            )

        counters["eligible"] = len(eligible)

        if args.balance_subjects:
            selected = choose_balanced_candidates(
                candidates=eligible,
                target_downloads=target_downloads,
                max_per_subject=args.max_per_subject,
            )
        else:
            selected = eligible[: target_downloads or None]

        counters["selected"] = len(selected)
        log(f"[POOL] eligible={len(eligible)} selected={len(selected)}")

        for item in selected:
            doc_id = item["doc_id"]
            book = item["book"]
            state = item["book_state"]
            slug = item["slug"]
            pubtime_dt = item["pubtime"]
            evidence = item["evidence"]
            pdf_url = item["pdf_url"]
            subjects = item["subjects"]

            resp = client.get(pdf_url, stream=False)
            if resp.status_code >= 400:
                counters["failed_pdf_download"] += 1
                append_jsonl(
                    paths.manifest_path,
                    {
                        "doc_id": doc_id,
                        "title": book.get("title", ""),
                        "status": "failed_pdf_download",
                        "pdf_url": pdf_url,
                        "http_status": resp.status_code,
                        "pubtime": pubtime_dt.isoformat(),
                        "evidence": [{"field": k, "value": v} for k, v in evidence],
                        "updated_at": now_iso(),
                    },
                )
                continue

            try:
                content = extract_pdf_text(resp.content)
            except Exception as exc:
                counters["failed_pdf_extract"] += 1
                append_jsonl(
                    paths.manifest_path,
                    {
                        "doc_id": doc_id,
                        "title": book.get("title", ""),
                        "status": "failed_pdf_extract",
                        "error": str(exc),
                        "pubtime": pubtime_dt.isoformat(),
                        "evidence": [{"field": k, "value": v} for k, v in evidence],
                        "updated_at": now_iso(),
                    },
                )
                continue

            if not content.strip():
                counters["failed_empty_text"] += 1
                append_jsonl(
                    paths.manifest_path,
                    {
                        "doc_id": doc_id,
                        "title": book.get("title", ""),
                        "status": "failed_empty_text",
                        "pubtime": pubtime_dt.isoformat(),
                        "evidence": [{"field": k, "value": v} for k, v in evidence],
                        "updated_at": now_iso(),
                    },
                )
                continue

            filename = safe_filename(doc_id) + ".txt"
            text_path = paths.text_dir / filename
            atomic_write_text(text_path, content)

            tokens = count_tokens(content)

            record = {
                "doc_id": doc_id,
                "source": SOURCE,
                "title": book.get("title", ""),
                "url": f"https://openstax.org/details/books/{slug}",
                "pdf_url": pdf_url,
                "slug": slug,
                "subjects": subjects,
                "book_state": state,
                "pubtime": pubtime_dt.isoformat(),
                "content": content,
                "tokens": tokens,
            }
            append_jsonl(paths.records_path, record)

            append_jsonl(
                paths.manifest_path,
                {
                    "doc_id": doc_id,
                    "title": book.get("title", ""),
                    "status": "downloaded",
                    "text_path": str(text_path),
                    "pubtime": pubtime_dt.isoformat(),
                    "evidence": [{"field": k, "value": v} for k, v in evidence],
                    "pdf_url": pdf_url,
                    "pdf_host": urlparse(pdf_url).netloc,
                    "tokens": tokens,
                    "updated_at": now_iso(),
                },
            )

            runtime_stats["file_count"] += 1
            runtime_stats["total_tokens"] += tokens
            runtime_stats["pubtime_distribution"][pubtime_dt.strftime("%Y-%m")] += 1
            for subject in subjects:
                runtime_stats["subject_distribution"][subject] += 1
            runtime_stats["subject_distribution_primary"][primary_subject(subjects)] += 1

            counters["downloaded"] += 1
            seen_ids.add(doc_id)
            save_progress(
                paths.progress_path,
                {
                    "updated_at": now_iso(),
                    "cutoff": cutoff_dt.isoformat(),
                    "done_ids": sorted(seen_ids),
                    "counts": dict(counters),
                },
            )
            log(f"[OK] {doc_id} tokens={tokens} pubtime={pubtime_dt.isoformat()}")
    finally:
        client.close()

    file_count = runtime_stats["file_count"]
    avg_tokens = (runtime_stats["total_tokens"] / file_count) if file_count else 0.0
    stats_payload = {
        "generated_at": now_iso(),
        "source": SOURCE,
        "cutoff": cutoff_dt.isoformat(),
        "counts": dict(counters),
        "file_count": file_count,
        "total_tokens": runtime_stats["total_tokens"],
        "avg_tokens": round(avg_tokens, 2),
        "pubtime_distribution": dict(sorted(runtime_stats["pubtime_distribution"].items())),
        "primary_subject_distribution": dict(
            sorted(runtime_stats["subject_distribution_primary"].items(), key=lambda x: x[1], reverse=True)
        ),
        "subject_distribution": dict(sorted(runtime_stats["subject_distribution"].items(), key=lambda x: x[1], reverse=True)),
    }
    atomic_write_json(paths.stats_path, stats_payload)
    print(json.dumps(stats_payload, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OpenStax text-only harvester with strict pubtime cutoff")
    sub = parser.add_subparsers(dest="command", required=True)

    p_harvest = sub.add_parser("harvest", help="Build OpenStax text corpus with pubtime gate")
    p_harvest.add_argument("--cutoff", required=True, help="Only include content with pubtime <= cutoff (ISO8601)")
    p_harvest.add_argument("--out", default="data/kb/openstax", help="Output root directory")
    p_harvest.add_argument("--rate", type=float, default=1.0, help="HTTP rate limit (requests/sec)")
    p_harvest.add_argument("--limit", type=int, default=0, help="Legacy alias: max selected books to download (0 means all)")
    p_harvest.add_argument("--target-downloads", type=int, default=0, help="Max selected books to download (0 means all)")
    p_harvest.add_argument("--balance-subjects", action="store_true", help="Enable round-robin balancing by primary subject")
    p_harvest.add_argument(
        "--max-per-subject",
        type=int,
        default=0,
        help="Max downloaded books per primary subject when --balance-subjects is enabled (0 means no cap)",
    )
    p_harvest.add_argument(
        "--subject-whitelist",
        default="",
        help="Comma-separated primary subjects to keep, e.g. 'Science,Social Sciences,Business'",
    )
    p_harvest.add_argument("--resume", action="store_true", help="Resume using saved progress state")
    p_harvest.set_defaults(func=harvest)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
