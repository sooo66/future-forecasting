"""Module base classes and run context."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Protocol

from core.io import SnapshotPaths


@dataclass(frozen=True)
class RunContext:
    snapshot_id: str
    date_from: datetime
    date_to: datetime
    resume: bool
    snapshot_paths: SnapshotPaths
    project_root: Path


class TextModule(Protocol):
    name: str

    def run(self, ctx: RunContext) -> Iterable[dict]:
        """Yield normalized TextRecord dicts."""

