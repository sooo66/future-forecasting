from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def _parse_iso(t: str) -> datetime | None:
    if not t:
        return None
    text = str(t).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _extract_offset_days(sampled: Dict[str, Any]) -> List[int]:
    vals: List[int] = []
    for k in sampled.keys():
        s = str(k).strip().lower()
        if not s.endswith("d"):
            continue
        n = s[:-1]
        if n.isdigit():
            vals.append(int(n))
    return sorted(set(vals), reverse=True)


def _horizon_status_counts(markets: List[Dict[str, Any]], offset_days: List[int]) -> Dict[int, Dict[str, int]]:
    counts: Dict[int, Dict[str, int]] = {d: {"covered": 0, "missing": 0} for d in offset_days}
    for m in markets:
        sampled = m.get("sampled", {})
        for d in offset_days:
            key = f"{d}d"
            obj = sampled.get(key) if isinstance(sampled, dict) else None
            p = obj.get("p_yes") if isinstance(obj, dict) else None
            if p is None:
                counts[d]["missing"] += 1
            else:
                counts[d]["covered"] += 1
    return counts


def make_visualizations(markets: List[Dict[str, Any]], output_dir: Path, ts: str) -> List[Path]:
    paths: List[Path] = []
    if plt is None:
        return paths

    domain_counts = Counter(m.get("domain", "other") for m in markets)
    if domain_counts:
        fig = plt.figure(figsize=(9, 4))
        xs = list(domain_counts.keys())
        ys = [domain_counts[x] for x in xs]
        plt.bar(xs, ys)
        plt.title("Domain Distribution")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        p = output_dir / f"q_domain_dist_{ts}.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)

    diff_counts = Counter()
    for m in markets:
        sampled = m.get("sampled", {})
        if not isinstance(sampled, dict):
            continue
        for obj in sampled.values():
            if isinstance(obj, dict):
                diff_counts[str(obj.get("difficult", "unknown"))] += 1
    if diff_counts:
        fig = plt.figure(figsize=(6, 4))
        order = ["easy", "medium", "hard", "unknown"]
        xs = [k for k in order if k in diff_counts] + [k for k in diff_counts if k not in order]
        ys = [diff_counts[x] for x in xs]
        plt.bar(xs, ys)
        plt.title("Difficult Distribution")
        plt.tight_layout()
        p = output_dir / f"q_difficult_dist_{ts}.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)

    by_month = Counter()
    for m in markets:
        dt = _parse_iso(str(m.get("resolve_time", "")))
        if dt is None:
            continue
        by_month[dt.strftime("%Y-%m")] += 1
    if by_month:
        fig = plt.figure(figsize=(11, 4))
        months = sorted(by_month.keys())
        vals = [by_month[m] for m in months]
        plt.plot(months, vals, marker="o", linewidth=1.5)
        plt.title("Resolve Time Distribution (Monthly)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        p = output_dir / f"q_resolve_time_dist_{ts}.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)

    sampled_keys = set()
    for m in markets:
        s = m.get("sampled", {})
        if isinstance(s, dict):
            sampled_keys.update(s.keys())
    offset_days = _extract_offset_days({k: 1 for k in sampled_keys})
    if offset_days:
        status = _horizon_status_counts(markets, offset_days)
        labels = [f"{d}d" for d in offset_days]
        covered = [status[d]["covered"] for d in offset_days]
        missing = [status[d]["missing"] for d in offset_days]

        fig = plt.figure(figsize=(8, 4))
        x = list(range(len(labels)))
        plt.bar(x, covered, label="covered")
        plt.bar(x, missing, bottom=covered, label="missing")
        plt.xticks(x, labels)
        plt.title("Sampled Horizon Availability")
        plt.legend()
        plt.tight_layout()
        p = output_dir / f"q_horizon_validity_{ts}.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)

    return paths


def make_visualizations_from_json(input_json: Path, output_dir: Path | None = None, ts: str | None = None) -> List[Path]:
    data = json.loads(input_json.read_text())
    markets = data.get("markets", [])
    if not isinstance(markets, list):
        raise ValueError("Invalid q.json: 'markets' must be a list")
    out_dir = output_dir or input_json.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = ts or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return make_visualizations(markets, out_dir, stamp)


def main() -> int:
    p = argparse.ArgumentParser(description="Generate visualizations from built q.json")
    p.add_argument("--input", required=True, type=str, help="Path to q_*.json")
    p.add_argument("--output-dir", type=str, default="", help="Directory for output PNGs (default: input dir)")
    p.add_argument("--ts", type=str, default="", help="Filename timestamp suffix (default: current UTC timestamp)")
    args = p.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.output_dir) if args.output_dir else None
    ts = args.ts or None
    paths = make_visualizations_from_json(in_path, out_dir, ts)
    print("visualizations=", [str(x) for x in paths])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
