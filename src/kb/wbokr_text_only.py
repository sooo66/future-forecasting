#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import tempfile
import time
import uuid
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode, urljoin, urlparse

import requests

OAI_BASE = "https://openknowledge.worldbank.org/server/oai/request"
PID_FIND_URL = "https://openknowledge.worldbank.org/server/api/pid/find"
BITSTREAM_CONTENT_TMPL = "https://openknowledge.worldbank.org/server/api/core/bitstreams/{uuid}/content"
SOURCE = "kb/world_bank"
UUID_RE = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
ITEM_HREF_RE = re.compile(r"https?://[^\s\"']+/server/api/core/items/([0-9a-fA-F-]{36})(?:\?.*)?$", re.IGNORECASE)
HANDLE_PATTERNS = [
    re.compile(r"hdl\.handle\.net/(10986/\d+)", re.IGNORECASE),
    re.compile(r"openknowledge\.worldbank\.org/handle/(10986/\d+)", re.IGNORECASE),
    re.compile(r"\b(10986/\d+)\b", re.IGNORECASE),
]


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

    def get(self, url: str, params: dict[str, Any] | None = None) -> requests.Response:
        retryable_codes = {429, 500, 502, 503, 504}
        backoff = 1.0
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            self._throttle()
            try:
                resp = self.session.get(
                    url,
                    params=params,
                    timeout=(10, 60),
                    allow_redirects=True,
                    headers={"Accept": "application/json, application/xml;q=0.9, */*;q=0.8"},
                )
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


class StateStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS processed_records (
                oai_identifier TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                doc_id TEXT,
                updated_at TEXT NOT NULL
            )
            """
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def has_processed(self, oai_identifier: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM processed_records WHERE oai_identifier = ?",
            (oai_identifier,),
        ).fetchone()
        return row is not None

    def mark_processed(self, oai_identifier: str, status: str, doc_id: str | None) -> None:
        self.conn.execute(
            """
            INSERT INTO processed_records (oai_identifier, status, doc_id, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(oai_identifier) DO UPDATE SET
                status = excluded.status,
                doc_id = excluded.doc_id,
                updated_at = excluded.updated_at
            """,
            (oai_identifier, status, doc_id, now_iso()),
        )
        self.conn.commit()


@dataclass
class HarvestPaths:
    root: Path
    metadata_dir: Path
    text_dir: Path
    state_dir: Path
    jsonl_path: Path
    progress_path: Path
    stats_path: Path
    sqlite_path: Path


@dataclass
class Counters:
    processed: int = 0
    downloaded: int = 0
    missing_handle: int = 0
    pid_resolve_failed: int = 0
    no_text_asset: int = 0
    text_download_failed: int = 0
    skipped_already_processed: int = 0
    failures: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "processed": self.processed,
            "downloaded": self.downloaded,
            "missing_handle": self.missing_handle,
            "pid_resolve_failed": self.pid_resolve_failed,
            "no_text_asset": self.no_text_asset,
            "text_download_failed": self.text_download_failed,
            "skipped_already_processed": self.skipped_already_processed,
            "failures": self.failures,
        }


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dirs(out_root: Path) -> HarvestPaths:
    metadata_dir = out_root / "metadata"
    text_dir = out_root / "text"
    state_dir = out_root / "state"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    return HarvestPaths(
        root=out_root,
        metadata_dir=metadata_dir,
        text_dir=text_dir,
        state_dir=state_dir,
        jsonl_path=metadata_dir / "okr_oai_records.jsonl",
        progress_path=state_dir / "progress.json",
        stats_path=metadata_dir / "stats.json",
        sqlite_path=state_dir / "harvest_state.sqlite",
    )


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


def load_progress(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def log(message: str) -> None:
    print(message, flush=True)


def save_progress(path: Path, from_date: str, until_date: str, last_token: str | None, counters: Counters) -> None:
    payload = {
        "from": from_date,
        "until": until_date,
        "last_resumption_token": last_token,
        "updated_at": now_iso(),
        "counts": counters.to_dict(),
    }
    atomic_write_json(path, payload)


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def token_count(text: str) -> int:
    # Approximation to avoid external tokenizer dependency.
    words = re.findall(r"\w+", text, flags=re.UNICODE)
    return len(words)


def parse_oai_xml(xml_text: str) -> tuple[list[dict[str, Any]], str | None]:
    ns = {
        "oai": "http://www.openarchives.org/OAI/2.0/",
        "dc": "http://purl.org/dc/elements/1.1/",
        "oai_dc": "http://www.openarchives.org/OAI/2.0/oai_dc/",
    }
    root = ET.fromstring(xml_text)
    records: list[dict[str, Any]] = []

    for rec in root.findall(".//oai:record", ns):
        header = rec.find("oai:header", ns)
        if header is None:
            continue
        if header.attrib.get("status") == "deleted":
            continue

        oai_identifier = safe_text(header.findtext("oai:identifier", default="", namespaces=ns))
        datestamp = safe_text(header.findtext("oai:datestamp", default="", namespaces=ns))
        meta = rec.find("oai:metadata", ns)

        dc_fields: dict[str, list[str]] = {
            "title": [],
            "creator": [],
            "subject": [],
            "description": [],
            "identifier": [],
            "relation": [],
            "language": [],
        }

        if meta is not None:
            dc_el = meta.find("oai_dc:dc", ns)
            if dc_el is not None:
                for key in dc_fields:
                    dc_fields[key] = [safe_text(e.text) for e in dc_el.findall(f"dc:{key}", ns) if safe_text(e.text)]

        records.append(
            {
                "oai_identifier": oai_identifier,
                "datestamp": datestamp,
                "dc": dc_fields,
            }
        )

    token_el = root.find(".//oai:resumptionToken", ns)
    token = safe_text(token_el.text if token_el is not None else "")
    return records, token or None


def extract_handle(candidates: list[str]) -> str | None:
    for pattern in HANDLE_PATTERNS:
        for text in candidates:
            m = pattern.search(text)
            if m:
                return m.group(1)
    return None


def extract_uuid_from_text(text: str) -> str | None:
    m = UUID_RE.search(text)
    if not m:
        return None
    return m.group(0).lower()


def detect_item_ref(payload: Any, text_blob: str) -> tuple[str | None, str | None]:
    item_uuid: str | None = None
    item_href: str | None = None

    def walk(node: Any) -> None:
        nonlocal item_uuid, item_href
        if item_uuid and item_href:
            return
        if isinstance(node, dict):
            for key, value in node.items():
                key_l = str(key).lower()
                if isinstance(value, str):
                    if "/api/core/items/" in value:
                        maybe = extract_uuid_from_text(value)
                        if maybe:
                            item_uuid = maybe
                        # Only accept canonical item endpoint, avoid subresources like /accessStatus.
                        m_item = ITEM_HREF_RE.match(value)
                        if m_item:
                            item_href = value
                    elif key_l in {"uuid", "id", "targetid", "dspaceobjectid"}:
                        maybe = extract_uuid_from_text(value)
                        if maybe:
                            item_uuid = maybe
                walk(value)
        elif isinstance(node, list):
            for i in node:
                walk(i)

    walk(payload)

    if not item_uuid:
        item_uuid = extract_uuid_from_text(text_blob)
    if not item_href and item_uuid:
        item_href = f"https://openknowledge.worldbank.org/server/api/core/items/{item_uuid}"

    return item_uuid, item_href


def get_json(client: HTTPClient, url: str, params: dict[str, Any] | None = None) -> tuple[dict[str, Any] | list[Any] | None, requests.Response]:
    resp = client.get(url, params=params)
    if resp.status_code >= 400:
        return None, resp
    try:
        return resp.json(), resp
    except ValueError:
        return None, resp


def resolve_item(client: HTTPClient, handle: str) -> tuple[str | None, dict[str, Any] | None]:
    payload, resp = get_json(client, PID_FIND_URL, params={"id": handle})
    blob = ""
    if payload is not None:
        blob = json.dumps(payload, ensure_ascii=False)
    blob += "\n" + resp.url
    location = resp.headers.get("Location", "")
    if location:
        blob += "\n" + location

    item_uuid, item_href = detect_item_ref(payload, blob)

    if not item_uuid and location:
        item_uuid = extract_uuid_from_text(location)

    if not item_href and item_uuid:
        item_href = f"https://openknowledge.worldbank.org/server/api/core/items/{item_uuid}"

    if not item_href and "/api/core/items/" in resp.url:
        item_href = resp.url
        if not item_uuid:
            item_uuid = extract_uuid_from_text(resp.url)

    if not item_href:
        return None, None

    item_payload, item_resp = get_json(client, item_href)
    if item_payload is None or item_resp.status_code >= 400:
        return None, None

    if not item_uuid:
        item_uuid = extract_uuid_from_text(json.dumps(item_payload, ensure_ascii=False))

    return item_uuid, item_payload if isinstance(item_payload, dict) else None


def get_link_href(item: dict[str, Any], key: str) -> str | None:
    links = item.get("_links", {})
    if not isinstance(links, dict):
        return None
    target = links.get(key)
    if isinstance(target, dict):
        href = target.get("href")
        if isinstance(href, str):
            return href
    return None


def fetch_hal_collection(client: HTTPClient, href: str, embedded_key: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    next_url = href
    while next_url:
        payload, resp = get_json(client, next_url)
        if payload is None or resp.status_code >= 400 or not isinstance(payload, dict):
            break
        embedded = payload.get("_embedded", {})
        page_items = embedded.get(embedded_key, []) if isinstance(embedded, dict) else []
        if isinstance(page_items, list):
            items.extend([x for x in page_items if isinstance(x, dict)])

        next_url = None
        links = payload.get("_links", {})
        if isinstance(links, dict):
            nxt = links.get("next")
            if isinstance(nxt, dict) and isinstance(nxt.get("href"), str):
                next_url = nxt["href"]

    return items


def bitstream_mime(client: HTTPClient, bitstream: dict[str, Any]) -> str:
    for key in ["mimeType", "mimetype", "mime", "type"]:
        value = bitstream.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    fmt = bitstream.get("format")
    if isinstance(fmt, dict):
        for key in ["mimeType", "mimetype"]:
            value = fmt.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip().lower()

    links = bitstream.get("_links", {})
    if isinstance(links, dict):
        fmt_link = links.get("format")
        if isinstance(fmt_link, dict) and isinstance(fmt_link.get("href"), str):
            payload, resp = get_json(client, fmt_link["href"])
            if payload is not None and resp.status_code < 400 and isinstance(payload, dict):
                for key in ["mimeType", "mimetype"]:
                    value = payload.get(key)
                    if isinstance(value, str) and value.strip():
                        return value.strip().lower()
    return ""


def select_text_bitstream(client: HTTPClient, bitstreams: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates: list[tuple[int, dict[str, Any]]] = []

    for bs in bitstreams:
        name = safe_text(bs.get("name", ""))
        name_l = name.lower()
        mime = bitstream_mime(client, bs)

        if mime == "application/pdf" or name_l.endswith(".pdf"):
            continue

        allow = mime.startswith("text/") or name_l.endswith(".txt")
        if not allow:
            continue

        score = 0
        if mime == "text/plain":
            score += 10
        if name_l.endswith(".txt"):
            score += 5
        size = bs.get("sizeBytes")
        if isinstance(size, int):
            score += max(0, min(size // 1024, 1000))
        candidates.append((score, bs))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def get_bitstreams_for_item(client: HTTPClient, item_payload: dict[str, Any]) -> list[dict[str, Any]]:
    bundles_href = get_link_href(item_payload, "bundles")
    if not bundles_href:
        return []

    bundles = fetch_hal_collection(client, bundles_href, "bundles")
    all_bitstreams: list[dict[str, Any]] = []

    for bundle in bundles:
        bitstreams_href = get_link_href(bundle, "bitstreams")
        if not bitstreams_href:
            b_uuid = safe_text(bundle.get("uuid", ""))
            if b_uuid:
                bitstreams_href = f"https://openknowledge.worldbank.org/server/api/core/bundles/{b_uuid}/bitstreams"
        if not bitstreams_href:
            continue
        all_bitstreams.extend(fetch_hal_collection(client, bitstreams_href, "bitstreams"))

    return all_bitstreams


def download_text_from_bitstream(client: HTTPClient, bitstream_uuid: str) -> str | None:
    url = BITSTREAM_CONTENT_TMPL.format(uuid=quote(bitstream_uuid))
    resp = client.get(url)
    if resp.status_code >= 400:
        return None
    try:
        return resp.content.decode("utf-8")
    except UnicodeDecodeError:
        return resp.content.decode("utf-8", errors="replace")


def parse_datestamp_month(datestamp: str) -> str:
    if not datestamp:
        return "unknown"
    try:
        if "T" in datestamp:
            dt = datetime.fromisoformat(datestamp.replace("Z", "+00:00"))
        else:
            dt = datetime.fromisoformat(datestamp)
        return f"{dt.year:04d}-{dt.month:02d}"
    except Exception:
        if len(datestamp) >= 7:
            return datestamp[:7]
        return "unknown"


def normalize_datestamp(datestamp: str) -> str:
    if not datestamp:
        return ""
    try:
        if "T" in datestamp:
            dt = datetime.fromisoformat(datestamp.replace("Z", "+00:00"))
        else:
            dt = datetime.fromisoformat(datestamp)
        return dt.date().isoformat()
    except Exception:
        return datestamp[:10]


def update_stats(stats: dict[str, Any], rec: dict[str, Any], tokens: int) -> None:
    stats["file_count"] += 1
    stats["total_tokens"] += tokens
    ym = parse_datestamp_month(rec.get("datestamp", ""))
    stats["time_distribution"][ym] += 1

    for sub in rec.get("subjects", []):
        if sub:
            stats["subject_distribution"][sub] += 1


def finalize_stats(stats: dict[str, Any]) -> dict[str, Any]:
    file_count = stats["file_count"]
    avg_tokens = (stats["total_tokens"] / file_count) if file_count else 0.0
    return {
        "file_count": file_count,
        "total_tokens": stats["total_tokens"],
        "avg_tokens": round(avg_tokens, 2),
        "time_distribution": dict(sorted(stats["time_distribution"].items())),
        "subject_distribution": dict(sorted(stats["subject_distribution"].items(), key=lambda x: x[1], reverse=True)),
        "generated_at": now_iso(),
    }


def harvest(args: argparse.Namespace) -> int:
    out_root = Path(args.out).resolve()
    paths = ensure_dirs(out_root)

    client = HTTPClient(rate=args.rate)
    state = StateStore(paths.sqlite_path)

    counters = Counters()
    runtime_stats = {
        "file_count": 0,
        "total_tokens": 0,
        "time_distribution": Counter(),
        "subject_distribution": Counter(),
    }

    progress = load_progress(paths.progress_path)
    last_token: str | None = None

    if args.resume and progress:
        if progress.get("from") == args.from_date and progress.get("until") == args.until_date:
            last_token = progress.get("last_resumption_token")
            old_counts = progress.get("counts", {})
            for key, value in old_counts.items():
                if hasattr(counters, key) and isinstance(value, int):
                    setattr(counters, key, value)

    log(
        f"[START] from={args.from_date} until={args.until_date} "
        f"out={paths.root} max_records={args.max_records} rate={args.rate} resume={args.resume}"
    )
    if last_token:
        log(f"[RESUME] last_resumption_token={str(last_token)[:60]}...")

    try:
        processed_limit = args.max_records
        page_no = 0
        while True:
            if processed_limit is not None and counters.processed >= processed_limit:
                log("[STOP] reached --max-records limit")
                break

            if last_token:
                params = {"verb": "ListRecords", "resumptionToken": last_token}
            else:
                params = {
                    "verb": "ListRecords",
                    "metadataPrefix": "oai_dc",
                    "from": args.from_date,
                    "until": args.until_date,
                }

            resp = client.get(OAI_BASE, params=params)
            if resp.status_code >= 400:
                counters.failures += 1
                save_progress(paths.progress_path, args.from_date, args.until_date, last_token, counters)
                print(f"[ERROR] OAI request failed: HTTP {resp.status_code}", file=sys.stderr)
                return 2

            try:
                oai_records, next_token = parse_oai_xml(resp.text)
            except ET.ParseError as exc:
                counters.failures += 1
                save_progress(paths.progress_path, args.from_date, args.until_date, last_token, counters)
                print(f"[ERROR] XML parse failed: {exc}", file=sys.stderr)
                return 2

            page_no += 1
            log(f"[OAI] page={page_no} records={len(oai_records)} next_token={'yes' if next_token else 'no'}")

            for oai in oai_records:
                if processed_limit is not None and counters.processed >= processed_limit:
                    break

                oai_identifier = oai.get("oai_identifier", "")
                if not oai_identifier:
                    counters.failures += 1
                    log("[WARN] missing oai_identifier")
                    continue

                if state.has_processed(oai_identifier):
                    counters.skipped_already_processed += 1
                    log(f"[SKIP] {oai_identifier}")
                    continue

                counters.processed += 1
                dc = oai.get("dc", {})
                identifiers = dc.get("identifier", []) + dc.get("relation", [])
                handle = extract_handle(identifiers)

                if not handle:
                    counters.missing_handle += 1
                    state.mark_processed(oai_identifier, "missing_handle", None)
                    save_progress(paths.progress_path, args.from_date, args.until_date, next_token, counters)
                    log(f"[MISS] handle: {oai_identifier}")
                    continue

                item_uuid, item_payload = resolve_item(client, handle)
                if not item_uuid or not item_payload:
                    counters.pid_resolve_failed += 1
                    counters.failures += 1
                    state.mark_processed(oai_identifier, "pid_resolve_failed", None)
                    save_progress(paths.progress_path, args.from_date, args.until_date, next_token, counters)
                    log(f"[FAIL] pid resolve: {oai_identifier} handle={handle}")
                    continue

                bitstreams = get_bitstreams_for_item(client, item_payload)
                selected = select_text_bitstream(client, bitstreams)
                if not selected:
                    counters.no_text_asset += 1
                    state.mark_processed(oai_identifier, "no_text_asset", item_uuid)
                    save_progress(paths.progress_path, args.from_date, args.until_date, next_token, counters)
                    log(f"[MISS] text asset: {oai_identifier} item={item_uuid}")
                    continue

                bs_uuid = safe_text(selected.get("uuid", ""))
                if not bs_uuid:
                    counters.text_download_failed += 1
                    counters.failures += 1
                    state.mark_processed(oai_identifier, "text_download_failed", item_uuid)
                    save_progress(paths.progress_path, args.from_date, args.until_date, next_token, counters)
                    log(f"[FAIL] bitstream uuid: {oai_identifier}")
                    continue

                content = download_text_from_bitstream(client, bs_uuid)
                if content is None:
                    counters.text_download_failed += 1
                    counters.failures += 1
                    state.mark_processed(oai_identifier, "text_download_failed", item_uuid)
                    save_progress(paths.progress_path, args.from_date, args.until_date, next_token, counters)
                    log(f"[FAIL] text download: {oai_identifier} bitstream={bs_uuid}")
                    continue

                if not content.strip():
                    counters.text_download_failed += 1
                    counters.failures += 1
                    state.mark_processed(oai_identifier, "text_download_failed", item_uuid)
                    save_progress(paths.progress_path, args.from_date, args.until_date, next_token, counters)
                    log(f"[FAIL] empty text: {oai_identifier} bitstream={bs_uuid}")
                    continue

                doc_id = item_uuid
                text_file = paths.text_dir / f"{doc_id}.txt"
                atomic_write_text(text_file, content)

                toks = token_count(content)
                record = {
                    "doc_id": doc_id,
                    "source": SOURCE,
                    "datestamp": normalize_datestamp(oai.get("datestamp", "")),
                    "url": f"https://openknowledge.worldbank.org/handle/{handle}",
                    "title": (dc.get("title", [""])[0] if dc.get("title") else ""),
                    "subjects": dc.get("subject", []) or [],
                    "description": "\n".join(dc.get("description", []) or []),
                    "language": (dc.get("language", [""])[0] if dc.get("language") else ""),
                    "content": content,
                    "tokens": toks,
                }
                append_jsonl(paths.jsonl_path, record)
                update_stats(runtime_stats, record, toks)

                counters.downloaded += 1
                state.mark_processed(oai_identifier, "downloaded", doc_id)
                save_progress(paths.progress_path, args.from_date, args.until_date, next_token, counters)
                log(f"[OK] doc_id={doc_id} tokens={toks} oai={oai_identifier}")

            last_token = next_token
            save_progress(paths.progress_path, args.from_date, args.until_date, last_token, counters)
            if not last_token:
                log("[DONE] no more resumptionToken")
                break

    finally:
        state.close()
        client.close()

    stats_payload = finalize_stats(runtime_stats)
    atomic_write_json(paths.stats_path, stats_payload)

    print(json.dumps({"counts": counters.to_dict(), "stats": stats_payload}, ensure_ascii=False, indent=2))
    return 0


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def stats_command(args: argparse.Namespace) -> int:
    out_root = Path(args.out).resolve()
    paths = ensure_dirs(out_root)
    rows = load_jsonl(paths.jsonl_path)

    stats = {
        "file_count": 0,
        "total_tokens": 0,
        "time_distribution": Counter(),
        "subject_distribution": Counter(),
    }

    for rec in rows:
        content = str(rec.get("content", ""))
        if not content.strip():
            continue
        tokens = rec.get("tokens")
        if not isinstance(tokens, int):
            tokens = token_count(content)
        update_stats(stats, rec, tokens)

    payload = finalize_stats(stats)
    atomic_write_json(paths.stats_path, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="World Bank OKR Text-Only Harvester (API-only, no HTML/PDF/OCR)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_harvest = sub.add_parser("harvest", help="Harvest records and text assets")
    p_harvest.add_argument("--from", dest="from_date", required=True, help="YYYY-MM-DD")
    p_harvest.add_argument("--until", dest="until_date", required=True, help="YYYY-MM-DD")
    p_harvest.add_argument("--out", default="data/kb/world_bank", help="Output root directory")
    p_harvest.add_argument("--max-records", type=int, default=20000, help="Max OAI records to process")
    p_harvest.add_argument("--rate", type=float, default=1.0, help="HTTP rate limit (requests/sec)")
    p_harvest.add_argument("--resume", action="store_true", help="Resume from progress state")
    p_harvest.set_defaults(func=harvest)

    p_stats = sub.add_parser("stats", help="Recompute stats from metadata jsonl")
    p_stats.add_argument("--out", default="data/kb/world_bank", help="Output root directory")
    p_stats.set_defaults(func=stats_command)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
