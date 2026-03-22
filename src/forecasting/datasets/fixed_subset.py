"""Load fixed question subsets from manifest files."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

from forecasting.contracts import QuestionRecord
from forecasting.datasets.questions import DEFAULT_HORIZON, load_questions


def load_fixed_question_subset(
    project_root: Path,
    subset_file: str,
) -> dict[str, Any]:
    manifest_path = (project_root / subset_file).resolve()
    manifest = _load_subset_manifest(manifest_path)
    source_path = (project_root / manifest["source_file"]).resolve()
    resolved_horizon = str(manifest.get("horizon") or DEFAULT_HORIZON)
    full_pool = load_questions(source_path, horizon_key=resolved_horizon)

    questions_by_id = {question["market_id"]: question for question in full_pool}
    subset: list[QuestionRecord] = []
    missing_ids: list[str] = []
    for market_id in manifest["market_ids"]:
        question = questions_by_id.get(market_id)
        if question is None:
            missing_ids.append(market_id)
            continue
        subset.append(dict(question))
    if missing_ids:
        raise ValueError(f"Missing market ids in fixed subset source {source_path}: {missing_ids}")

    stats = _build_fixed_subset_stats(
        subset,
        source_pool_size=len(full_pool),
        subset_id=str(manifest.get("subset_id") or manifest_path.stem),
        selection_strategy=str(manifest.get("selection_strategy") or "fixed subset manifest"),
    )
    _validate_manifest_stats(manifest.get("stats"), stats)

    return {
        "subset_id": str(manifest.get("subset_id") or manifest_path.stem),
        "description": str(manifest.get("description") or "").strip(),
        "subset_file": str(manifest_path),
        "source_file": str(source_path),
        "horizon": resolved_horizon,
        "market_ids": list(manifest["market_ids"]),
        "questions": subset,
        "stats": stats,
    }


def _load_subset_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected fixed subset payload shape in {path}")

    source_file = payload.get("source_file")
    market_ids = payload.get("market_ids")
    if not isinstance(source_file, str) or not source_file.strip():
        raise ValueError(f"Fixed subset manifest {path} must include a non-empty source_file")
    if not isinstance(market_ids, list) or not market_ids:
        raise ValueError(f"Fixed subset manifest {path} must include a non-empty market_ids list")

    normalized_ids = [str(item).strip() for item in market_ids]
    if any(not item for item in normalized_ids):
        raise ValueError(f"Fixed subset manifest {path} contains empty market ids")
    if len(set(normalized_ids)) != len(normalized_ids):
        raise ValueError(f"Fixed subset manifest {path} contains duplicate market ids")

    return {
        **payload,
        "source_file": source_file.strip(),
        "market_ids": normalized_ids,
    }


def _build_fixed_subset_stats(
    subset: list[QuestionRecord],
    *,
    source_pool_size: int,
    subset_id: str,
    selection_strategy: str,
) -> dict[str, Any]:
    return {
        "selected_count": len(subset),
        "source_pool_size": int(source_pool_size),
        "domains": dict(sorted(Counter(question["domain"] for question in subset).items())),
        "difficulties": dict(sorted(Counter(question["difficulty"] for question in subset).items())),
        "sampling": {
            "strategy": "fixed subset manifest",
            "subset_id": subset_id,
            "selection_strategy": selection_strategy,
        },
    }


def _validate_manifest_stats(expected: Any, actual: dict[str, Any]) -> None:
    if not isinstance(expected, dict):
        return
    for key in ("selected_count", "domains", "difficulties"):
        if key in expected and expected[key] != actual[key]:
            raise ValueError(
                f"Fixed subset manifest stats mismatch for {key}: expected {expected[key]!r}, got {actual[key]!r}"
            )
