"""Per-module resume state (seen record ids)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ResumeState:
    def __init__(self, path: Path, *, enable_resume: bool) -> None:
        self.path = path
        self.enable_resume = enable_resume
        self._seen: set[str] = set()
        if enable_resume:
            self._load()

    @property
    def seen(self) -> set[str]:
        return self._seen

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        ids = raw.get("seen_ids", [])
        if isinstance(ids, list):
            self._seen = {str(x) for x in ids if str(x)}

    def has(self, record_id: str) -> bool:
        return record_id in self._seen

    def add(self, record_id: str) -> None:
        self._seen.add(record_id)

    def save(self, extra: dict[str, Any] | None = None) -> None:
        payload = {
            "seen_ids": sorted(self._seen),
        }
        if extra:
            payload.update(extra)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

