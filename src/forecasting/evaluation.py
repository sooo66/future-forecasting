"""Evaluation helpers for forecasting experiments."""

from __future__ import annotations

import json
from collections import Counter
from statistics import mean
from typing import Any

from forecasting.contracts import ForecastResult


def compute_ece(
    predictions: list[float],
    labels: list[int],
    *,
    bins: int = 10,
) -> tuple[float, list[dict[str, Any]]]:
    if not predictions or not labels or len(predictions) != len(labels):
        return 0.0, []
    total = len(predictions)
    bucket_rows: list[dict[str, Any]] = []
    ece = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        if index == bins - 1:
            members = [i for i, value in enumerate(predictions) if lower <= value <= upper]
        else:
            members = [i for i, value in enumerate(predictions) if lower <= value < upper]
        if not members:
            continue
        avg_pred = sum(predictions[i] for i in members) / len(members)
        avg_label = sum(labels[i] for i in members) / len(members)
        gap = abs(avg_label - avg_pred)
        ece += (len(members) / total) * gap
        bucket_rows.append(
            {
                "bin_index": index,
                "lower": lower,
                "upper": upper,
                "count": len(members),
                "avg_pred": avg_pred,
                "avg_label": avg_label,
                "abs_gap": gap,
            }
        )
    return ece, bucket_rows


def summarize_results(results: list[ForecastResult]) -> dict[str, Any]:
    total_tokens = [item["total_tokens"] for item in results if item.get("total_tokens") is not None]
    step_counts = [int(item["steps_count"]) for item in results if item.get("steps_count") is not None]
    method_names = [str(item.get("method_name") or "").strip() for item in results if str(item.get("method_name") or "").strip()]
    source_counter = Counter()
    tool_counter = Counter()
    predictions = [float(item["predicted_prob"]) for item in results]
    labels = [int(item["label"]) for item in results]
    ece_10, calibration_bins = compute_ece(predictions, labels, bins=10)
    for item in results:
        source_counter.update(item.get("retrieved_source_types") or [])
        tool_counter.update(item.get("tool_usage_counts") or {})
    return {
        "display_name": method_names[0] if method_names and len(set(method_names)) == 1 else None,
        "count": len(results),
        "accuracy": mean(int((float(item["predicted_prob"]) >= 0.5) == bool(int(item["label"]))) for item in results) if results else 0.0,
        "brier_score": mean(item["brier_score"] for item in results) if results else 0.0,
        "ece_10": ece_10,
        "avg_latency_sec": mean(item["latency_sec"] for item in results) if results else 0.0,
        "avg_total_tokens": mean(total_tokens) if total_tokens else None,
        "avg_steps_count": mean(step_counts) if step_counts else None,
        "token_usage_coverage": (len(total_tokens) / len(results)) if results else 0.0,
        "steps_count_coverage": (len(step_counts) / len(results)) if results else 0.0,
        "retrieved_source_type_counts": dict(sorted(source_counter.items())),
        "tool_usage_counts": dict(sorted(tool_counter.items())),
        "calibration_bins": calibration_bins,
    }


def render_experiment_summary(summary: dict[str, Any]) -> str:
    settings = summary.get("settings") or {}
    dataset = settings.get("dataset") or {}
    methods = summary.get("methods") or {}
    ordered_method_ids = list(settings.get("method_ids") or methods.keys())

    rows: list[str] = []
    for method_id in ordered_method_ids:
        metrics = methods.get(method_id)
        if not isinstance(metrics, dict):
            continue
        display_name = str(metrics.get("display_name") or method_id)
        avg_total_tokens = metrics.get("avg_total_tokens")
        avg_steps_count = metrics.get("avg_steps_count")
        rows.append(
            f"| {display_name} | {metrics.get('count', 0)} | {float(metrics.get('accuracy') or 0.0):.3f} | "
            f"{float(metrics.get('brier_score') or 0.0):.4f} | {float(metrics.get('ece_10') or 0.0):.4f} | "
            f"{float(metrics.get('avg_latency_sec') or 0.0):.2f} | "
            f"{avg_total_tokens if avg_total_tokens is not None else 'n/a'} | "
            f"{f'{float(avg_steps_count):.1f}' if avg_steps_count is not None else 'n/a'} |"
        )
    if not rows:
        rows.append("| n/a | 0 | 0.000 | 0.0000 | 0.0000 | 0.00 | n/a | n/a |")

    return (
        f"# Experiment Summary: {summary.get('experiment_id', 'unknown')}\n\n"
        "## Runtime\n\n"
        f"- Model: `{summary.get('model')}`\n"
        f"- Base URL: `{summary.get('base_url')}`\n"
        f"- Subset size: `{summary.get('subset_size')}`\n"
        f"- Horizon: `{summary.get('horizon')}`\n"
        f"- Knowledge root: `{settings.get('knowledge_root')}`\n"
        f"- Search API base: `{settings.get('search_api_base')}`\n"
        f"- Search snapshot root: `{settings.get('search_snapshot_root')}`\n"
        f"- Max parallel methods: `{settings.get('max_parallel_methods')}`\n\n"
        "## Dataset\n\n"
        f"- Dataset file: `{dataset.get('dataset_file')}`\n"
        f"- Subset id: `{dataset.get('subset_id')}`\n"
        f"- Horizon: `{dataset.get('horizon')}`\n\n"
        "## Method Configs\n\n"
        "```json\n"
        f"{json.dumps(settings.get('method_configs') or {}, ensure_ascii=False, indent=2)}\n"
        "```\n\n"
        "## Results\n\n"
        "| Method | Count | Accuracy | Brier | ECE@10 | Avg latency (s) | Avg total tokens | Avg steps |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        f"{chr(10).join(rows)}\n"
    )


def score_tuple(metrics: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(metrics.get("brier_score") or 0.0),
        float(metrics.get("ece_10") or 0.0),
        -float(metrics.get("accuracy") or 0.0),
    )
