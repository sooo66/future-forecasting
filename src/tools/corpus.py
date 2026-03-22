"""Offline passage corpus builder for search."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
from typing import Any, Iterable

import tiktoken

from utils.time_utils import to_day


DEFAULT_CHUNK_TOKENS = 512
DEFAULT_CHUNK_OVERLAP = 128
DEFAULT_TOKENIZER_NAME = "cl100k_base"


def build_corpus(
    snapshot_root: Path,
    output_path: Path,
    *,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    tokenizer_name: str = DEFAULT_TOKENIZER_NAME,
) -> dict[str, Any]:
    snapshot_root = Path(snapshot_root).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_tokens:
        raise ValueError("chunk_overlap must be in [0, chunk_tokens)")

    encoder = tiktoken.get_encoding(tokenizer_name)
    rows: list[dict[str, Any]] = []
    doc_count = 0
    source_counter: Counter[str] = Counter()

    for path in _iter_record_paths(snapshot_root):
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                passage_rows = _to_passage_rows(
                    row,
                    encoder=encoder,
                    chunk_tokens=chunk_tokens,
                    chunk_overlap=chunk_overlap,
                )
                if not passage_rows:
                    continue
                rows.extend(passage_rows)
                doc_count += 1
                source_counter.update(item["source"] for item in passage_rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False))
            fh.write("\n")

    stats = {
        "snapshot_root": str(snapshot_root),
        "corpus_path": str(output_path),
        "doc_count": doc_count,
        "passage_count": len(rows),
        "chunk_tokens": int(chunk_tokens),
        "chunk_overlap": int(chunk_overlap),
        "tokenizer": tokenizer_name,
        "source_counts": dict(sorted(source_counter.items())),
    }
    stats_path = output_path.with_name("stats.json")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def _iter_record_paths(snapshot_root: Path) -> list[Path]:
    return sorted(path for path in snapshot_root.rglob("records.jsonl") if path.is_file())


def _to_passage_rows(
    row: dict[str, Any],
    *,
    encoder: tiktoken.Encoding,
    chunk_tokens: int,
    chunk_overlap: int,
) -> list[dict[str, Any]]:
    payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
    doc_id = str(row.get("id") or "").strip()
    kind = str(row.get("kind") or "").strip().lower()
    raw_source = str(row.get("source") or "").strip().lower()
    timestamp = _normalize_day(row.get("timestamp"))
    if not doc_id or not kind or not raw_source or not timestamp:
        return []

    source_type = raw_source.split("/", 1)[0] if "/" in raw_source else raw_source
    coarse_source = f"{kind}/{source_type}"
    title = _normalize_title(payload, source_type)
    body = _build_body(payload, source_type)
    url = str(row.get("url")).strip() if row.get("url") else None

    chunks = _chunk_text(body, encoder=encoder, chunk_tokens=chunk_tokens, chunk_overlap=chunk_overlap)
    if not chunks:
        fallback = _clean_text(body or title)
        if fallback:
            chunks = [fallback]
        elif title:
            chunks = [title]
        else:
            return []

    out: list[dict[str, Any]] = []
    for idx, chunk in enumerate(chunks):
        content = _clean_text(chunk or title)
        if not content:
            continue
        passage_id = f"{doc_id}::chunk{idx:04d}"
        out.append(
            {
                "id": passage_id,
                "doc_id": doc_id,
                "source": coarse_source,
                "timestamp": timestamp,
                "title": title,
                "content": content,
                "contents": _compose_contents(title, content),
                "url": url,
            }
        )
    return out


def _compose_contents(title: str, content: str) -> str:
    title_text = _clean_text(title)
    content_text = _clean_text(content)
    if title_text and content_text:
        return f"\"{title_text}\"\n{content_text}"
    return title_text or content_text


def _build_body(payload: dict[str, Any], source_type: str) -> str:
    description = _normalize_description(payload, source_type)
    content = _normalize_main_content(payload)
    parts: list[str] = []

    if source_type == "sociomedia":
        if description:
            parts.append(description)
        if content:
            parts.append(content)
        comments = payload.get("comments")
        if isinstance(comments, list):
            joined_comments = "\n".join(_clean_text(item) for item in comments if _clean_text(item))
            if joined_comments:
                parts.append(joined_comments)
        return "\n".join(part for part in parts if part).strip()

    if description:
        parts.append(description)
    if content:
        parts.append(content)
    return "\n".join(part for part in parts if part).strip()


def _normalize_title(payload: dict[str, Any], source_type: str) -> str:
    title = _clean_text(payload.get("title"))
    if title:
        return title
    if source_type == "book":
        return _clean_text(payload.get("page_title"))
    if source_type == "sociomedia":
        return _clean_text(payload.get("subreddit"))
    return ""


def _normalize_description(payload: dict[str, Any], source_type: str) -> str:
    parts: list[str] = []
    if source_type == "blog":
        author = _clean_text(payload.get("author"))
        if author:
            parts.append(f"author: {author}")
    if source_type == "paper":
        authors = payload.get("authors")
        if isinstance(authors, list) and authors:
            parts.append("authors: " + ", ".join(_clean_text(x) for x in authors if _clean_text(x)))
    if source_type == "sociomedia":
        subreddit = _clean_text(payload.get("subreddit"))
        score = payload.get("score")
        num_comments = payload.get("num_comments")
        meta = " ".join(
            part
            for part in [
                f"subreddit: {subreddit}" if subreddit else "",
                f"score: {score}" if score is not None else "",
                f"num_comments: {num_comments}" if num_comments is not None else "",
            ]
            if part
        )
        if meta:
            parts.append(meta)
    if source_type == "book":
        page_title = _clean_text(payload.get("page_title"))
        title = _clean_text(payload.get("title"))
        if page_title and page_title != title:
            parts.append(page_title)

    description = _clean_text(payload.get("description"))
    if description:
        parts.append(description)
    return _clean_text(" ".join(parts))


def _normalize_main_content(payload: dict[str, Any]) -> str:
    content = payload.get("content")
    if isinstance(content, list):
        content = "\n".join(_clean_text(item) for item in content if _clean_text(item))
    return _clean_text(content)


def _chunk_text(
    text: str,
    *,
    encoder: tiktoken.Encoding,
    chunk_tokens: int,
    chunk_overlap: int,
) -> list[str]:
    cleaned = _clean_text(text)
    if not cleaned:
        return []
    token_ids = encoder.encode(cleaned)
    if len(token_ids) <= chunk_tokens:
        return [cleaned]

    out: list[str] = []
    start = 0
    while start < len(token_ids):
        end = min(len(token_ids), start + chunk_tokens)
        chunk = encoder.decode(token_ids[start:end]).strip()
        if chunk:
            out.append(_clean_text(chunk))
        if end >= len(token_ids):
            break
        start = max(end - chunk_overlap, start + 1)
    return out


def _normalize_day(value: Any) -> str | None:
    day = to_day(value)
    if day:
        return day
    text = str(value or "").strip()
    if not text:
        return None
    raise ValueError(f"Invalid date/time value: {value}")


def _clean_text(value: Any) -> str:
    text = str(value or "")
    return re.sub(r"\s+", " ", text).strip()
