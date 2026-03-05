"""World Bank report KB module adapter into unified text schema."""
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


class WorldBankModule:
    name = "kb.report.world_bank"

    def __init__(
        self,
        *,
        harvest_rate: float = 5.0,
        max_records: int = 30000,
    ) -> None:
        self.harvest_rate = max(0.1, float(harvest_rate))
        self.max_records = max(1, int(max_records))

    def _build_harvest_cmd(
        self,
        ctx: RunContext,
        *,
        out_dir: Path,
        from_day: str,
        to_day: str,
        resume: bool,
        max_records: int,
    ) -> List[str]:
        cmd = [
            sys.executable,
            str(ctx.project_root / "src" / "modules" / "kb" / "harvesters" / "worldbank_harvester.py"),
            "harvest",
            "--from",
            from_day,
            "--to",
            to_day,
            "--max-records",
            str(max_records),
            "--rate",
            str(self.harvest_rate),
            "--out",
            str(out_dir),
        ]
        if resume:
            cmd.append("--resume")
        return cmd

    def _normalize_row(self, row: dict, *, from_day: str, to_day_str: str) -> Optional[dict]:
        ts = to_iso_utc(row.get("datestamp")) or row.get("datestamp")
        day = to_day(ts)
        if not day:
            return None
        if day < from_day or day > to_day_str:
            return None
        payload = {
            "title": str(row.get("title") or ""),
            "content": row.get("content") or "",
        }
        source = "report/world_bank"
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
        from_day: str,
        to_day_str: str,
    ) -> Iterator[dict]:
        records_path = work_dir / "metadata" / "okr_oai_records.jsonl"
        log_path = work_dir / "metadata" / "harvest.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = self._build_harvest_cmd(
            ctx,
            out_dir=work_dir,
            from_day=from_day,
            to_day=to_day_str,
            resume=ctx.resume,
            max_records=self.max_records,
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
                            normalized = self._normalize_row(row, from_day=from_day, to_day_str=to_day_str)
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
                        normalized = self._normalize_row(row, from_day=from_day, to_day_str=to_day_str)
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
        logger.info(f"[{self.name}] harvesting world bank from={ctx.date_from.date()} to={ctx.date_to.date()}")
        work_dir = ctx.snapshot_paths.module_work_dir(self.name)
        from_day = ctx.date_from.date().isoformat()
        to_day_str = ctx.date_to.date().isoformat()

        def _iter():
            count = 0
            for row in self._stream_harvest_records(
                ctx,
                work_dir=work_dir,
                from_day=from_day,
                to_day_str=to_day_str,
            ):
                count += 1
                yield row
            logger.info(f"[{self.name}] normalized records={count}")

        return _iter()
