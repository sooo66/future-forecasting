#!/usr/bin/env python3
from __future__ import annotations

import argparse
from bs4 import BeautifulSoup
import json
import os
import re
import tempfile
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any

import requests

BOOKS_API_URL = "https://openstax.org/apps/cms/api/books/?format=json"
BOOK_DETAIL_API_TMPL = "https://openstax.org/apps/cms/api/books/{slug}/?format=json"
SITEMAP_URL = "https://openstax.org/sitemap.xml"
SOURCE = "kb/openstax"


@dataclass
class HarvestPaths:
    root: Path
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
    metadata_dir = out_root / "metadata"
    state_dir = out_root / "state"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    return HarvestPaths(
        root=out_root,
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
        self.session.trust_env = False
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


def get_book_detail(client: HTTPClient, slug: str) -> dict[str, Any] | None:
    url = BOOK_DETAIL_API_TMPL.format(slug=slug)
    resp = client.get(url)
    resp.encoding = "utf-8"
    if resp.status_code >= 400:
        return None
    try:
        payload = resp.json()
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def parse_preloaded_state(html_text: str) -> dict[str, Any] | None:
    match = re.search(r"window\.__PRELOADED_STATE__\s*=\s*(\{.*?\})\s*</script>", html_text, flags=re.S)
    if not match:
        return None
    raw = match.group(1).strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def resolve_archive_config(client: HTTPClient, rex_page_url: str) -> tuple[str | None, dict[str, dict[str, Any]]]:
    resp = client.get(rex_page_url)
    resp.encoding = "utf-8"
    if resp.status_code >= 400:
        return None, {}
    state = parse_preloaded_state(resp.text)
    if not state:
        return None, {}

    content = state.get("content") if isinstance(state.get("content"), dict) else {}
    book = content.get("book") if isinstance(content.get("book"), dict) else {}
    load_options = book.get("loadOptions") if isinstance(book.get("loadOptions"), dict) else {}
    books_cfg = load_options.get("booksConfig") if isinstance(load_options.get("booksConfig"), dict) else {}
    archive_url = str(books_cfg.get("archiveUrl") or "").strip()
    if archive_url.startswith("/"):
        archive_url = f"https://openstax.org{archive_url}"
    elif archive_url and not archive_url.startswith("http://") and not archive_url.startswith("https://"):
        archive_url = ""
    books_map = books_cfg.get("books") if isinstance(books_cfg.get("books"), dict) else {}
    return archive_url or None, books_map


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


def primary_subject(subjects: list[str]) -> str:
    if subjects:
        return subjects[0]
    return "Unknown"


def parse_csv_list(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def get_archive_book_toc_html(client: HTTPClient, archive_base: str, book_ref: str) -> str:
    url = f"{archive_base}/contents/{book_ref}.json"
    resp = client.get(url)
    resp.encoding = "utf-8"
    if resp.status_code >= 400:
        return ""
    try:
        payload = resp.json()
    except ValueError:
        return ""
    if not isinstance(payload, dict):
        return ""
    return str(payload.get("content") or "")


def extract_page_refs_from_toc(toc_html: str) -> list[str]:
    if not toc_html.strip():
        return []
    soup = BeautifulSoup(toc_html, "lxml")
    refs: list[str] = []
    seen: set[str] = set()
    for a in soup.find_all("a"):
        href = str(a.get("href") or "").strip()
        if not href.startswith("./") or not href.endswith(".xhtml"):
            continue
        ref = href[2:]
        if ref in seen:
            continue
        seen.add(ref)
        refs.append(ref)
    return refs


def fetch_archive_page_xhtml(client: HTTPClient, archive_base: str, page_ref: str) -> str:
    url = f"{archive_base}/contents/{page_ref}"
    resp = client.get(url)
    resp.encoding = "utf-8"
    if resp.status_code >= 400:
        return ""
    return resp.text


def extract_page_content_text(xhtml: str) -> tuple[str, str]:
    if not xhtml.strip():
        return "", ""
    soup = BeautifulSoup(xhtml, "lxml")
    for tag in soup.find_all(["script", "style", "noscript"]):
        tag.decompose()
    title_text = ""
    title_node = soup.find("title")
    if title_node:
        title_text = " ".join(title_node.get_text(" ", strip=True).split())
    body = soup.find("body")
    node = body if body else soup
    text = node.get_text("\n", strip=True)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return title_text, text


def choose_pubtime(catalog_book: dict[str, Any], detail_book: dict[str, Any], details_lastmod: str | None) -> tuple[datetime | None, list[tuple[str, str]]]:
    evidence: list[tuple[str, str]] = []
    candidates: list[datetime] = []

    detail_updated = detail_book.get("updated")
    if isinstance(detail_updated, str) and detail_updated.strip():
        dt = parse_iso_datetime(detail_updated)
        if dt:
            candidates.append(dt)
            evidence.append(("detail.updated", detail_updated))

    publish_date = detail_book.get("publish_date") or catalog_book.get("publish_date")
    if isinstance(publish_date, str) and publish_date.strip():
        dt = parse_iso_datetime(publish_date)
        if dt:
            candidates.append(dt)
            evidence.append(("publish_date", publish_date))

    last_updated_pdf = detail_book.get("last_updated_pdf") or catalog_book.get("last_updated_pdf")
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
    return max(candidates), evidence


def choose_balanced_candidates(
    candidates: list[dict[str, Any]],
    target_records: int,
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
            if target_records > 0 and len(selected) >= target_records:
                return selected
        if not progressed:
            break

    return selected


def count_tokens(text: str) -> int:
    return len(re.findall(r"\w+", text, flags=re.UNICODE))


def log(msg: str) -> None:
    print(msg, flush=True)


def harvest(args: argparse.Namespace) -> int:
    to_value = (args.to_date or args.cutoff or "").strip()
    cutoff_dt = parse_cutoff(to_value) if to_value else None
    from_dt = parse_iso_datetime(args.from_date) if args.from_date else None
    if from_dt and cutoff_dt and from_dt > cutoff_dt:
        raise ValueError("--from must be <= --to")
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
    seen_books = set(progress.get("done_books", [])) if isinstance(progress.get("done_books"), list) else set()

    try:
        books = get_books_catalog(client)
        counters["books_total"] = len(books)
        details_lastmod_map = get_details_lastmod_map(client)

        target_records = args.target_records if args.target_records > 0 else args.target_downloads
        if target_records <= 0:
            target_records = args.limit
        subject_whitelist = set(parse_csv_list(args.subject_whitelist)) if args.subject_whitelist else set()

        live_books = [b for b in books if str(b.get("book_state", "")).strip().lower() == "live"]
        counters["books_live_total"] = len(live_books)
        if not live_books:
            raise RuntimeError("No live books available from OpenStax books API")

        archive_base = ""
        archive_book_versions: dict[str, dict[str, Any]] = {}
        for book in live_books:
            slug = normalize_slug(str(book.get("slug") or "").strip())
            if not slug:
                continue
            detail = get_book_detail(client, slug)
            if not detail:
                continue
            rex_url = str(detail.get("webview_rex_link") or "").strip()
            if not rex_url:
                continue
            resolved_archive, resolved_versions = resolve_archive_config(client, rex_url)
            if resolved_archive and resolved_versions:
                archive_base = resolved_archive
                archive_book_versions = resolved_versions
                break
        if not archive_base or not archive_book_versions:
            raise RuntimeError("Failed to resolve OpenStax archive config from REX page")

        log(
            f"[START] from={from_dt.isoformat() if from_dt else 'none'} to={cutoff_dt.isoformat() if cutoff_dt else 'none'} "
            f"out={paths.root} books_total={len(books)} "
            f"books_live_total={len(live_books)} archive={archive_base} "
            f"target_records={target_records or 'all'} balanced={args.balance_subjects} "
            f"max_per_subject={args.max_per_subject or 'none'} resume={args.resume}"
        )

        eligible: list[dict[str, Any]] = []
        for book in live_books:
            state = "live"
            slug_raw = str(book.get("slug", "")).strip()
            slug = normalize_slug(slug_raw)
            if not slug:
                counters["skipped_missing_slug"] += 1
                continue

            detail_book = get_book_detail(client, slug)
            if not detail_book:
                counters["skipped_no_detail_api"] += 1
                continue

            book_uuid = str(detail_book.get("book_uuid") or detail_book.get("cnx_id") or "").strip()
            if not book_uuid:
                counters["skipped_missing_book_uuid"] += 1
                continue

            version_entry = archive_book_versions.get(book_uuid)
            default_version = ""
            if isinstance(version_entry, dict):
                default_version = str(version_entry.get("defaultVersion") or "").strip()
            if not default_version:
                counters["skipped_missing_archive_version"] += 1
                append_jsonl(
                    paths.manifest_path,
                    {
                        "doc_id": f"openstax_{slug.replace('/', '_')}",
                        "title": detail_book.get("title") or book.get("title") or "",
                        "status": "skipped_missing_archive_version",
                        "book_uuid": book_uuid,
                        "updated_at": now_iso(),
                    },
                )
                continue

            book_ref = f"{book_uuid}@{default_version}"
            book_key = f"{slug}|{book_ref}"
            if book_key in seen_books:
                counters["skipped_already_done"] += 1
                continue

            details_lastmod = details_lastmod_map.get(slug)
            pubtime_dt, evidence = choose_pubtime(book, detail_book, details_lastmod)

            if pubtime_dt is None:
                counters["skipped_no_pubtime"] += 1
                append_jsonl(
                    paths.manifest_path,
                    {
                        "doc_id": f"openstax_{slug.replace('/', '_')}",
                        "title": book.get("title", ""),
                        "status": "skipped_no_pubtime",
                        "evidence": [{"field": k, "value": v} for k, v in evidence],
                        "updated_at": now_iso(),
                    },
                )
                continue

            if from_dt and pubtime_dt < from_dt:
                counters["skipped_before_from"] += 1
                append_jsonl(
                    paths.manifest_path,
                    {
                        "doc_id": f"openstax_{slug.replace('/', '_')}",
                        "title": book.get("title", ""),
                        "status": "skipped_before_from",
                        "pubtime": pubtime_dt.isoformat(),
                        "from": from_dt.isoformat(),
                        "evidence": [{"field": k, "value": v} for k, v in evidence],
                        "updated_at": now_iso(),
                    },
                )
                continue

            if cutoff_dt and pubtime_dt > cutoff_dt:
                counters["skipped_after_cutoff"] += 1
                append_jsonl(
                    paths.manifest_path,
                    {
                        "doc_id": f"openstax_{slug.replace('/', '_')}",
                        "title": book.get("title", ""),
                        "status": "skipped_after_cutoff",
                        "pubtime": pubtime_dt.isoformat(),
                        "cutoff": cutoff_dt.isoformat() if cutoff_dt else None,
                        "evidence": [{"field": k, "value": v} for k, v in evidence],
                        "updated_at": now_iso(),
                    },
                )
                continue

            subjects = detail_book.get("book_subjects") if isinstance(detail_book.get("book_subjects"), list) else []
            if not subjects:
                subjects = book.get("subjects") if isinstance(book.get("subjects"), list) else []
            if not subjects:
                subjects = book.get("subject_categories") if isinstance(book.get("subject_categories"), list) else []
            if not subjects:
                subjects = book.get("k12subject") if isinstance(book.get("k12subject"), list) else []
            subjects = [str(x).strip() for x in subjects if isinstance(x, str) and str(x).strip()]
            book_primary_subject = primary_subject(subjects)
            if subject_whitelist and book_primary_subject not in subject_whitelist:
                counters["skipped_subject_not_selected"] += 1
                append_jsonl(
                    paths.manifest_path,
                    {
                        "doc_id": f"openstax_{slug.replace('/', '_')}",
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
                    "doc_id": f"openstax_{slug.replace('/', '_')}",
                    "title": str(detail_book.get("title") or book.get("title") or ""),
                    "slug": slug,
                    "book_uuid": book_uuid,
                    "book_ref": book_ref,
                    "book_key": book_key,
                    "detail_book": detail_book,
                    "book_state": state,
                    "book": book,
                    "pubtime": pubtime_dt,
                    "evidence": evidence,
                    "subjects": subjects,
                    "primary_subject": book_primary_subject,
                }
            )

        counters["eligible"] = len(eligible)

        if args.balance_subjects:
            selected = choose_balanced_candidates(
                candidates=eligible,
                target_records=target_records,
                max_per_subject=args.max_per_subject,
            )
        else:
            selected = eligible[: target_records or None]

        counters["selected"] = len(selected)
        log(f"[POOL] eligible={len(eligible)} selected={len(selected)}")

        for item in selected:
            book = item["book"]
            detail_book = item["detail_book"]
            state = "live"
            slug = str(item["slug"])
            pubtime_dt = item["pubtime"]
            evidence = item["evidence"]
            subjects = item["subjects"]
            book_ref = str(item["book_ref"])
            book_key = str(item["book_key"])
            book_title = str(detail_book.get("title") or book.get("title") or "").strip()
            book_doc_id = str(item["doc_id"])

            toc_html = get_archive_book_toc_html(client, archive_base, book_ref)
            if not toc_html.strip():
                counters["skipped_no_archive_toc"] += 1
                append_jsonl(
                    paths.manifest_path,
                    {
                        "doc_id": book_doc_id,
                        "title": book_title,
                        "status": "skipped_no_archive_toc",
                        "book_ref": book_ref,
                        "pubtime": pubtime_dt.isoformat(),
                        "updated_at": now_iso(),
                    },
                )
                continue

            page_refs = extract_page_refs_from_toc(toc_html)
            if not page_refs:
                counters["skipped_empty_toc"] += 1
                append_jsonl(
                    paths.manifest_path,
                    {
                        "doc_id": book_doc_id,
                        "title": book_title,
                        "status": "skipped_empty_toc",
                        "book_ref": book_ref,
                        "pubtime": pubtime_dt.isoformat(),
                        "updated_at": now_iso(),
                    },
                )
                continue

            harvested_pages = 0
            skipped_empty_page = 0
            for page_ref in page_refs:
                xhtml = fetch_archive_page_xhtml(client, archive_base, page_ref)
                if not xhtml.strip():
                    counters["page_fetch_failed"] += 1
                    continue

                page_title, page_text = extract_page_content_text(xhtml)
                if not page_text.strip():
                    skipped_empty_page += 1
                    counters["page_empty_text"] += 1
                    continue

                page_hash = hashlib.sha1(page_ref.encode("utf-8")).hexdigest()[:16]
                page_doc_id = f"{book_doc_id}_{page_hash}"
                page_url = f"{archive_base}/contents/{page_ref}"
                final_title = page_title or book_title or slug
                content = "\n".join(
                    [
                        f"Book: {book_title or slug}",
                        f"Page: {final_title}",
                        "",
                        page_text,
                    ]
                ).strip()
                tokens = count_tokens(content)

                record = {
                    "doc_id": page_doc_id,
                    "source": SOURCE,
                    "title": final_title,
                    "book_title": book_title,
                    "page_title": page_title or final_title,
                    "url": page_url,
                    "slug": slug,
                    "book_ref": book_ref,
                    "page_ref": page_ref,
                    "subjects": subjects,
                    "book_state": state,
                    "pubtime": pubtime_dt.isoformat(),
                    "content": content,
                    "tokens": tokens,
                }
                append_jsonl(paths.records_path, record)

                runtime_stats["file_count"] += 1
                runtime_stats["total_tokens"] += tokens
                runtime_stats["pubtime_distribution"][pubtime_dt.strftime("%Y-%m")] += 1
                for subject in subjects:
                    runtime_stats["subject_distribution"][subject] += 1
                runtime_stats["subject_distribution_primary"][primary_subject(subjects)] += 1
                counters["pages_harvested"] += 1
                harvested_pages += 1

            if harvested_pages <= 0:
                counters["skipped_no_page_content"] += 1
                append_jsonl(
                    paths.manifest_path,
                    {
                        "doc_id": book_doc_id,
                        "title": book_title,
                        "status": "skipped_no_page_content",
                        "book_ref": book_ref,
                        "page_refs": len(page_refs),
                        "pubtime": pubtime_dt.isoformat(),
                        "evidence": [{"field": k, "value": v} for k, v in evidence],
                        "updated_at": now_iso(),
                    },
                )
                continue

            counters["harvested"] += 1
            counters["downloaded"] += 1
            counters["books_harvested"] += 1
            seen_books.add(book_key)
            append_jsonl(
                paths.manifest_path,
                {
                    "doc_id": book_doc_id,
                    "title": book_title,
                    "status": "harvested_archive_pages",
                    "book_ref": book_ref,
                    "pages_total": len(page_refs),
                    "pages_harvested": harvested_pages,
                    "pages_skipped_empty": skipped_empty_page,
                    "pubtime": pubtime_dt.isoformat(),
                    "evidence": [{"field": k, "value": v} for k, v in evidence],
                    "updated_at": now_iso(),
                },
            )
            save_progress(
                paths.progress_path,
                {
                    "updated_at": now_iso(),
                    "from": from_dt.isoformat() if from_dt else None,
                    "to": cutoff_dt.isoformat() if cutoff_dt else None,
                    "cutoff": cutoff_dt.isoformat() if cutoff_dt else None,
                    "done_books": sorted(seen_books),
                    "counts": dict(counters),
                },
            )
            log(f"[OK] {book_doc_id} pages={harvested_pages}/{len(page_refs)} pubtime={pubtime_dt.isoformat()} mode=archive_xhtml")
    finally:
        client.close()

    file_count = runtime_stats["file_count"]
    avg_tokens = (runtime_stats["total_tokens"] / file_count) if file_count else 0.0
    stats_payload = {
        "generated_at": now_iso(),
        "source": SOURCE,
        "from": from_dt.isoformat() if from_dt else None,
        "to": cutoff_dt.isoformat() if cutoff_dt else None,
        "cutoff": cutoff_dt.isoformat() if cutoff_dt else None,
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
    parser = argparse.ArgumentParser(description="OpenStax live books harvester using archive XHTML API")
    sub = parser.add_subparsers(dest="command", required=True)

    p_harvest = sub.add_parser("harvest", help="Build OpenStax live corpus from archive XHTML pages")
    p_harvest.add_argument("--from", dest="from_date", default="", help="Only include content with pubtime >= from (ISO8601)")
    p_harvest.add_argument("--to", dest="to_date", default="", help="Only include content with pubtime <= to (ISO8601)")
    p_harvest.add_argument("--cutoff", default="", help="Deprecated alias of --to")
    p_harvest.add_argument("--out", default="data/kb/openstax", help="Output root directory")
    p_harvest.add_argument("--rate", type=float, default=1.0, help="HTTP rate limit (requests/sec)")
    p_harvest.add_argument("--limit", type=int, default=0, help="Legacy alias: max selected books to harvest (0 means all)")
    p_harvest.add_argument("--target-records", type=int, default=0, help="Max selected books to harvest (0 means all)")
    p_harvest.add_argument("--target-downloads", type=int, default=0, help="Deprecated alias of --target-records")
    p_harvest.add_argument("--balance-subjects", action="store_true", help="Enable round-robin balancing by primary subject")
    p_harvest.add_argument(
        "--max-per-subject",
        type=int,
        default=0,
        help="Max harvested books per primary subject when --balance-subjects is enabled (0 means no cap)",
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
    if getattr(args, "command", None) == "harvest":
        args.to_date = (getattr(args, "to_date", "") or "").strip()
        args.cutoff = (getattr(args, "cutoff", "") or "").strip()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
