"""Typed experiment specifications."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    dataset_file: str
    knowledge_root: str
    method_ids: list[str]
    method_configs: dict[str, Any]
    output_dir: str
    max_parallel_methods: int = 1
