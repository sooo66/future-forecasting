"""Substack blog module adapter into unified text schema."""
from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import threading
import time
from typing import Iterable, Iterator, List, Optional

from loguru import logger

from core.contracts import TextRecord, stable_record_id
from modules.base import RunContext
from modules.info.blog.substack_importer import load_import_config, SubstackImporter
from utils.time_utils import to_day


def _fmt_iso(dt: datetime, *, is_end: bool) -> str:
    d = dt.astimezone(timezone.utc).date().isoformat()
    return f"{d}T23:59:59+00:00" if is_end else f"{d}T00:00:00+00:00"


def _iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


class SubstackModule:
    name = "info.blog.substack"

    def __init__(
        self,
    ) -> None:
        pass

    @staticmethod
    def _normalize_row(row: dict, *, from_day: str, to_day_str: str) -> Optional[dict]:
        source_raw = str(row.get("source") or "substack/unknown")
        source_tail = source_raw.split("/", 1)[1] if "/" in source_raw else source_raw
        source = "blog/substack"
        title = str(row.get("title") or "")
        content = row.get("content") or ""
        description = row.get("description")
        day = row.get("timestamp") or to_day(row.get("published_at")) or to_day(row.get("pubtime"))
        if not day:
            return None
        if day < from_day or day > to_day_str:
            return None
        author = str(row.get("author") or source_tail or "").strip() or None
        payload = {
            "title": title,
            "author": author,
            "description": description,
            "content": content,
        }
        rid = stable_record_id(source, row.get("url"), day, title)
        return TextRecord(
            id=rid,
            kind="info",
            source=source,
            timestamp=day,
            url=row.get("url"),
            payload=payload,
        ).normalized().to_dict()

    def _stream_import(
        self,
        *,
        importer: SubstackImporter,
        output_file: Path,
        from_day: str,
        to_day_str: str,
    ) -> Iterator[dict]:
        worker_error: dict[str, Exception] = {}

        def _worker():
            try:
                importer.run()
            except Exception as exc:
                worker_error["exc"] = exc

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

        yielded_ids: set[str] = set()
        offset = 0

        while True:
            if output_file.exists():
                with output_file.open("r", encoding="utf-8") as f:
                    f.seek(offset)
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if not isinstance(row, dict):
                            continue
                        normalized = self._normalize_row(row, from_day=from_day, to_day_str=to_day_str)
                        if not normalized:
                            continue
                        rid = str(normalized.get("id") or "")
                        if not rid or rid in yielded_ids:
                            continue
                        yielded_ids.add(rid)
                        yield normalized
                    offset = f.tell()

            if not t.is_alive():
                break
            time.sleep(0.2)

        t.join()
        if output_file.exists():
            with output_file.open("r", encoding="utf-8") as f:
                f.seek(offset)
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(row, dict):
                        continue
                    normalized = self._normalize_row(row, from_day=from_day, to_day_str=to_day_str)
                    if not normalized:
                        continue
                    rid = str(normalized.get("id") or "")
                    if not rid or rid in yielded_ids:
                        continue
                    yielded_ids.add(rid)
                    yield normalized

        if "exc" in worker_error:
            raise worker_error["exc"]

    def _build_importer(
        self,
        *,
        cfg_path: str,
        work_dir: Path,
        date_from: datetime,
        date_to: datetime,
    ) -> tuple[SubstackImporter, Path]:
        cfg = load_import_config(cfg_path)
        output_dir = work_dir / "output"
        state_path = work_dir / "state.json"

        start_iso = _fmt_iso(date_from, is_end=False)
        end_iso = _fmt_iso(date_to, is_end=True)
        authors = [replace(author, start_date=start_iso, end_date=end_iso) for author in cfg.authors if author.enabled]

        run_cfg = replace(cfg, output_dir=output_dir, state_path=state_path, authors=authors)
        importer = SubstackImporter(run_cfg)
        return importer, output_dir / "substack_all.jsonl"

    def run(self, ctx: RunContext) -> Iterable[dict]:
        logger.info(f"[{self.name}] importing substack from={ctx.date_from.date()} to={ctx.date_to.date()}")
        work_dir = ctx.snapshot_paths.module_work_dir(self.name)
        cfg_path = "config/substack_authors.toml"
        importer, out_file = self._build_importer(
            cfg_path=cfg_path,
            work_dir=work_dir,
            date_from=ctx.date_from,
            date_to=ctx.date_to,
        )
        from_day = ctx.date_from.date().isoformat()
        to_day_str = ctx.date_to.date().isoformat()

        def _iter():
            count = 0
            for row in self._stream_import(
                importer=importer,
                output_file=out_file,
                from_day=from_day,
                to_day_str=to_day_str,
            ):
                count += 1
                yield row
            logger.info(f"[{self.name}] normalized records={count}")
            if count == 0:
                logger.warning(f"[{self.name}] no records emitted; output_file={out_file}")

        return _iter()
