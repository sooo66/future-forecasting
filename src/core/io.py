"""Filesystem layout and writers for benchmark snapshots."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


MODULE_OUTPUT_PATHS = {
    "info.news": "info/news/records.jsonl",
    "info.blog.substack": "info/blog/substack/records.jsonl",
    "info.sociomedia.reddit": "info/sociomedia/reddit/records.jsonl",
    "info.paper.arxiv": "info/paper/arxiv/records.jsonl",
    "kb.book.openstax": "kb/book/openstax/records.jsonl",
    "kb.report.world_bank": "kb/report/world_bank/records.jsonl",
}


@dataclass(frozen=True)
class SnapshotPaths:
    root: Path
    meta_dir: Path
    state_dir: Path
    work_dir: Path

    @classmethod
    def create(cls, snapshot_root: Path) -> "SnapshotPaths":
        root = snapshot_root
        meta_dir = root / "_meta"
        state_dir = root / "_state"
        work_dir = root / "_work"
        meta_dir.mkdir(parents=True, exist_ok=True)
        state_dir.mkdir(parents=True, exist_ok=True)
        work_dir.mkdir(parents=True, exist_ok=True)
        return cls(root=root, meta_dir=meta_dir, state_dir=state_dir, work_dir=work_dir)

    def records_path(self, module_name: str) -> Path:
        rel = MODULE_OUTPUT_PATHS.get(module_name)
        if not rel:
            raise KeyError(f"Unknown module output mapping: {module_name}")
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def state_path(self, module_name: str) -> Path:
        path = self.state_dir / f"{module_name.replace('.', '_')}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def module_work_dir(self, module_name: str) -> Path:
        path = self.work_dir / module_name.replace(".", "/")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def run_manifest_path(self) -> Path:
        return self.meta_dir / "run_manifest.json"

    def stats_path(self) -> Path:
        return self.meta_dir / "stats.json"


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False))
        f.write("\n")

