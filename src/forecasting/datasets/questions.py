"""Question loading and reproducible subset sampling."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import random
from typing import Any

from forecasting.contracts import QuestionRecord
from utils.time_utils import parse_datetime, to_iso_utc


DEFAULT_HORIZON = "7d"


@dataclass(frozen=True)
class BucketSpec:
    label: str
    start: str
    end: str


def load_questions(path: Path, *, horizon_key: str = DEFAULT_HORIZON) -> list[QuestionRecord]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    markets = payload.get("markets") if isinstance(payload, dict) else None
    if not isinstance(markets, list):
        raise ValueError(f"Unexpected question payload shape in {path}")

    out: list[QuestionRecord] = []
    for row in markets:
        if not isinstance(row, dict):
            continue
        sampled = row.get("sampled")
        horizon_sample = sampled.get(horizon_key) if isinstance(sampled, dict) else None
        if not isinstance(horizon_sample, dict):
            continue
        label = row.get("answer")
        if label not in {0, 1}:
            continue
        open_time = to_iso_utc(row.get("open_time"))
        resolve_time = to_iso_utc(row.get("resolve_time"))
        if not open_time or not resolve_time:
            continue
        out.append(
            {
                "market_id": str(row.get("market_id") or ""),
                "question": str(row.get("question") or "").strip(),
                "description": str(row.get("description") or "").strip(),
                "resolution_criteria": str(row.get("resolution_criteria") or "").strip(),
                "domain": str(row.get("domain") or "other").strip() or "other",
                "open_time": open_time,
                "resolve_time": resolve_time,
                "resolved_time": resolve_time,
                "sample_time": to_iso_utc(horizon_sample.get("t")),
                "difficulty": str(horizon_sample.get("difficult") or "unknown").strip() or "unknown",
                "sampled_prob_yes": float(horizon_sample.get("p_yes") or 0.0),
                "label": int(label),
                "horizon": horizon_key,
            }
        )

    out.sort(key=lambda item: (item["resolve_time"], item["open_time"], item["market_id"]))
    return out


def select_question_subset(
    questions: list[QuestionRecord],
    *,
    target_size: int,
    seed: int,
    time_bucket_count: int = 8,
) -> tuple[list[QuestionRecord], dict[str, Any]]:
    if target_size <= 0:
        raise ValueError("target_size must be positive")
    if target_size >= len(questions):
        return list(questions), build_subset_stats(questions, questions, [], [])

    enriched = [dict(question) for question in questions]
    open_specs = _assign_time_buckets(enriched, "open_time", "open_bucket", time_bucket_count)
    resolve_specs = _assign_time_buckets(enriched, "resolve_time", "resolve_bucket", time_bucket_count)

    domain_targets = _sqrt_targets(Counter(q["domain"] for q in enriched), target_size)
    difficulty_targets = _uniform_targets(sorted({q["difficulty"] for q in enriched}), target_size)
    open_targets = _uniform_targets([spec.label for spec in open_specs], target_size)
    resolve_targets = _uniform_targets([spec.label for spec in resolve_specs], target_size)

    rng = random.Random(seed)
    counts = {
        "domain": Counter(),
        "difficulty": Counter(),
        "open_bucket": Counter(),
        "resolve_bucket": Counter(),
    }
    selected_ids: set[str] = set()
    selected: list[QuestionRecord] = []

    def choose_best(predicate) -> dict[str, Any] | None:
        if len(selected) >= target_size:
            return None
        candidates = [q for q in enriched if q["market_id"] not in selected_ids and predicate(q)]
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda q: (
                _selection_gain(q, counts, domain_targets, difficulty_targets, open_targets, resolve_targets),
                rng.random(),
            ),
        )

    for value in sorted(domain_targets, key=lambda item: (domain_targets[item], item)):
        if len(selected) >= target_size:
            break
        candidate = choose_best(lambda q, value=value: q["domain"] == value)
        if candidate:
            _select(candidate, selected, selected_ids, counts)
    for value in sorted(difficulty_targets):
        if len(selected) >= target_size:
            break
        candidate = choose_best(lambda q, value=value: q["difficulty"] == value)
        if candidate:
            _select(candidate, selected, selected_ids, counts)
    for value in sorted(open_targets):
        if len(selected) >= target_size:
            break
        candidate = choose_best(lambda q, value=value: q["open_bucket"] == value)
        if candidate:
            _select(candidate, selected, selected_ids, counts)
    for value in sorted(resolve_targets):
        if len(selected) >= target_size:
            break
        candidate = choose_best(lambda q, value=value: q["resolve_bucket"] == value)
        if candidate:
            _select(candidate, selected, selected_ids, counts)

    while len(selected) < target_size:
        remaining = [q for q in enriched if q["market_id"] not in selected_ids]
        if not remaining:
            break
        best = max(
            remaining,
            key=lambda q: (
                _selection_gain(q, counts, domain_targets, difficulty_targets, open_targets, resolve_targets),
                rng.random(),
            ),
        )
        _select(best, selected, selected_ids, counts)

    selected.sort(key=lambda item: (item["resolve_time"], item["open_time"], item["market_id"]))
    stats = build_subset_stats(selected, questions, open_specs, resolve_specs)
    stats["sampling"] = {
        "strategy": "coverage-seeded greedy balancing over open_time bucket, resolve_time bucket, domain, difficulty",
        "seed": seed,
        "target_size": target_size,
        "bucketed": True,
        "time_bucket_count": time_bucket_count,
        "targets": {
            "domain": dict(domain_targets),
            "difficulty": dict(difficulty_targets),
            "open_bucket": dict(open_targets),
            "resolve_bucket": dict(resolve_targets),
        },
    }
    return selected, stats


def build_subset_stats(
    subset: list[QuestionRecord],
    full_pool: list[QuestionRecord],
    open_specs: list[BucketSpec],
    resolve_specs: list[BucketSpec],
) -> dict[str, Any]:
    def _range(values: list[str]) -> dict[str, str]:
        if not values:
            return {"start": "", "end": ""}
        return {"start": min(values), "end": max(values)}

    return {
        "selected_count": len(subset),
        "available_count": len(full_pool),
        "domains": dict(sorted(Counter(item["domain"] for item in subset).items())),
        "difficulties": dict(sorted(Counter(item["difficulty"] for item in subset).items())),
        "open_time_range": _range([item["open_time"] for item in subset]),
        "resolve_time_range": _range([item["resolve_time"] for item in subset]),
        "open_bucket_distribution": dict(sorted(Counter(item.get("open_bucket") for item in subset).items())),
        "resolve_bucket_distribution": dict(sorted(Counter(item.get("resolve_bucket") for item in subset).items())),
        "open_bucket_specs": [spec.__dict__ for spec in open_specs],
        "resolve_bucket_specs": [spec.__dict__ for spec in resolve_specs],
    }


def _selection_gain(
    question: dict[str, Any],
    counts: dict[str, Counter],
    domain_targets: dict[str, int],
    difficulty_targets: dict[str, int],
    open_targets: dict[str, int],
    resolve_targets: dict[str, int],
) -> float:
    weights = {
        "open_bucket": 3.0,
        "resolve_bucket": 3.0,
        "difficulty": 2.0,
        "domain": 1.5,
    }
    targets = {
        "domain": domain_targets,
        "difficulty": difficulty_targets,
        "open_bucket": open_targets,
        "resolve_bucket": resolve_targets,
    }
    score = 0.0
    for key, weight in weights.items():
        value = question[key]
        deficit = max(0, int(targets[key].get(value, 0)) - int(counts[key].get(value, 0)))
        score += weight * deficit
        score += 0.05 / (1 + int(counts[key].get(value, 0)))
    return score


def _select(
    question: dict[str, Any],
    selected: list[QuestionRecord],
    selected_ids: set[str],
    counts: dict[str, Counter],
) -> None:
    if question["market_id"] in selected_ids:
        return
    selected_ids.add(question["market_id"])
    selected.append(question)
    counts["domain"][question["domain"]] += 1
    counts["difficulty"][question["difficulty"]] += 1
    counts["open_bucket"][question["open_bucket"]] += 1
    counts["resolve_bucket"][question["resolve_bucket"]] += 1


def _uniform_targets(values: list[str], target_size: int) -> dict[str, int]:
    unique_values = sorted({value for value in values if value})
    if not unique_values:
        return {}
    base = target_size // len(unique_values)
    remainder = target_size % len(unique_values)
    out = {value: base for value in unique_values}
    for value in unique_values[:remainder]:
        out[value] += 1
    return out


def _assign_time_buckets(
    questions: list[dict[str, Any]],
    time_key: str,
    output_key: str,
    bucket_count: int,
) -> list[BucketSpec]:
    items = sorted(
        (
            parse_datetime(question[time_key]) or datetime.min,
            idx,
            question[time_key],
        )
        for idx, question in enumerate(questions)
    )
    if not items:
        return []

    bucket_count = max(1, min(bucket_count, len(items)))
    bucket_to_values: defaultdict[int, list[str]] = defaultdict(list)
    for position, (_dt, idx, original) in enumerate(items):
        bucket_index = min(bucket_count - 1, (position * bucket_count) // len(items))
        label = f"{time_key}:{bucket_index + 1:02d}"
        questions[idx][output_key] = label
        bucket_to_values[bucket_index].append(original)

    specs: list[BucketSpec] = []
    for bucket_index in range(bucket_count):
        values = bucket_to_values[bucket_index]
        specs.append(
            BucketSpec(
                label=f"{time_key}:{bucket_index + 1:02d}",
                start=min(values),
                end=max(values),
            )
        )
    return specs


def _sqrt_targets(counter: Counter[str], target_size: int) -> dict[str, int]:
    if not counter:
        return {}
    weights = {key: math.sqrt(value) for key, value in counter.items()}
    total_weight = sum(weights.values()) or 1.0
    raw = {key: (weight / total_weight) * target_size for key, weight in weights.items()}
    out = {key: max(1, int(math.floor(value))) for key, value in raw.items()}
    shortfall = target_size - sum(out.values())
    if shortfall > 0:
        for key, _ in sorted(raw.items(), key=lambda item: (item[1] - math.floor(item[1]), item[0]), reverse=True)[:shortfall]:
            out[key] += 1
    elif shortfall < 0:
        for key, _ in sorted(raw.items(), key=lambda item: (item[1], item[0]), reverse=True):
            if shortfall == 0:
                break
            if out[key] > 1:
                out[key] -= 1
                shortfall += 1
    return dict(sorted(out.items()))
