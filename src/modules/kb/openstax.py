"""OpenStax KB module adapter into unified text schema."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Iterable, Iterator, List, Optional

from loguru import logger

from core.contracts import TextRecord, stable_record_id
from modules.base import RunContext
from utils.time_utils import to_day, to_iso_utc

class OpenStaxModule:
    name = "kb.book.openstax"

    def __init__(
        self,
        *,
        harvest_rate: float = 3.0,
    ) -> None:
        self.harvest_rate = max(0.1, float(harvest_rate))

    def _build_harvest_cmd(
        self,
        ctx: RunContext,
        *,
        out_dir: Path,
        resume: bool,
    ) -> List[str]:
        cmd = [
            sys.executable,
            str(ctx.project_root / "src" / "modules" / "kb" / "harvesters" / "openstax_harvester.py"),
            "harvest",
            "--rate",
            str(self.harvest_rate),
            "--out",
            str(out_dir),
        ]
        if resume:
            cmd.append("--resume")
        return cmd

    def _normalize_row(self, row: dict) -> Optional[dict]:
        ts = to_iso_utc(row.get("pubtime")) or row.get("pubtime")
        day = to_day(ts)
        if not day:
            return None
        source = "book/openstax"
        payload = {
            "title": str(row.get("title") or ""),
            "book_title": str(row.get("book_title") or ""),
            "page_title": str(row.get("page_title") or ""),
            "subjects": row.get("subjects") or [],
            "content": row.get("content") or "",
        }
        rid = stable_record_id(source, row.get("url"), day, payload["title"])
        return TextRecord(
            id=rid,
            kind="kb",
            source=source,
            timestamp=day,
            url=row.get("url"),
            payload=payload,
        ).normalized().to_dict()

    def _stream_harvest_records(
        self,
        ctx: RunContext,
        *,
        work_dir: Path,
    ) -> Iterator[dict]:
        records_path = work_dir / "metadata" / "openstax_records.jsonl"
        log_path = work_dir / "metadata" / "harvest.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = self._build_harvest_cmd(
            ctx,
            out_dir=work_dir,
            resume=ctx.resume,
        )

        yielded_ids: set[str] = set()
        offset = 0
        with log_path.open("w", encoding="utf-8") as lf:
            proc = subprocess.Popen(
                cmd,
                cwd=str(ctx.project_root),
                stdout=lf,
                stderr=subprocess.STDOUT,
                text=True,
            )

            while True:
                if records_path.exists():
                    with records_path.open("r", encoding="utf-8") as f:
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
                            normalized = self._normalize_row(row)
                            if not normalized:
                                continue
                            rid = str(normalized.get("id") or "")
                            if not rid or rid in yielded_ids:
                                continue
                            yielded_ids.add(rid)
                            yield normalized
                        offset = f.tell()

                rc = proc.poll()
                if rc is not None:
                    break
                time.sleep(0.2)

            if records_path.exists():
                with records_path.open("r", encoding="utf-8") as f:
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
                        normalized = self._normalize_row(row)
                        if not normalized:
                            continue
                        rid = str(normalized.get("id") or "")
                        if not rid or rid in yielded_ids:
                            continue
                        yielded_ids.add(rid)
                        yield normalized
            if proc.returncode != 0:
                tail = ""
                try:
                    tail = "\n".join(log_path.read_text(encoding="utf-8").splitlines()[-20:])
                except OSError:
                    tail = ""
                logger.error(f"[{self.name}] harvest failed with returncode={proc.returncode}\n{tail}")

    def run(self, ctx: RunContext) -> Iterable[dict]:
        logger.info(f"[{self.name}] harvesting openstax live corpus (date window ignored)")
        work_dir = ctx.snapshot_paths.module_work_dir(self.name)

        def _iter():
            count = 0
            for row in self._stream_harvest_records(
                ctx,
                work_dir=work_dir,
            ):
                count += 1
                yield row
            logger.info(f"[{self.name}] normalized records={count}")

        return _iter()
