#!/usr/bin/env python3
"""
Build resolved binary forecasting dataset from official Polymarket APIs.

Official docs used:
- Gamma Markets API (paginated): GET /markets
  https://docs.polymarket.com/api-reference/markets/list-markets
- Gamma Tags API: GET /tags/slug/{slug}
  https://docs.polymarket.com/api-reference/tags/get-tag-by-slug
- CLOB price history: GET /prices-history
  https://docs.polymarket.com/api-reference/pricing/get-price-history-for-a-traded-token
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import requests

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

from question_visualization import make_visualizations


GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"

# ---- defaults ----
DEFAULT_PAGE_LIMIT = 80
DEFAULT_MIN_VOLUME_TOTAL = 100.0
DEFAULT_VALIDATE_N = 20
DEFAULT_SAMPLE_OFFSETS_DAYS = [30, 7, 1]
DEFAULT_MIN_DURATION_DAYS = 1
DEFAULT_RECENT_RESOLVE_DAYS = 120
DEFAULT_MAX_RECENT_RESOLVE_DAYS = 365
DEFAULT_EXPAND_RESOLVE_STEP_DAYS = 45
DEFAULT_MAX_SERIES_POINTS = 180
DEFAULT_MAX_RUNTIME_SEC = 900
DEFAULT_MAX_EMPTY_PAGES = 40
DEFAULT_PANEL_DOMAIN_QUOTA = 40
DEFAULT_PANEL_MIN_COUNTS = "30:80,7:150,1:150"
DEFAULT_MIN_ACTIVE_DOMAINS = 2
DEFAULT_MIN_PER_DOMAIN = 15
DEFAULT_MAX_DOMAIN_RATIO = 4.0
DEFAULT_MAX_POOL_SIZE = 5000
DEFAULT_MIN_FINAL_COUNT = 900
DEFAULT_MAX_FINAL_COUNT = 1200
DEFAULT_TARGET_POOL_RATIO = 0.22
DEFAULT_SELECTION_STABILITY_PATIENCE = 32
DEFAULT_DOMAIN_TARGET_SLOTS = 12
DEFAULT_DIVERSITY_PENALTY = 6.0
DEFAULT_STRICT_MODE = True
DEFAULT_MATERIALIZE_BATCH_SIZE = 40

REQUEST_TIMEOUT = 8
MAX_RETRIES = 2
BASE_BACKOFF_SEC = 0.6
RATE_LIMIT_SLEEP_SEC = 0.03
RANDOM_SEED = 42

GENERIC_TAGS = {
    "all",
    "events",
    "event",
    "games",
    "hide-from-new",
    "mention-markets",
    "recurring",
}

DOMAIN_TAG_SLUGS: Dict[str, List[str]] = {
    "politics": ["politics", "elections", "current-events"],
    "crypto": ["crypto"],
    "finance": ["finance", "equities", "stocks", "business"],
    "tech": ["technology", "ai"],
    "culture": ["pop-culture", "entertainment"],
    "weather": ["weather"],
    "world": ["world"],
    "sports": ["sports"],
}


@dataclass
class Config:
    page_limit: int
    min_volume_total: float
    validate_n: int
    sample_offsets_days: List[int]
    min_duration_days: int
    recent_resolve_days: int
    max_recent_resolve_days: int
    expand_resolve_step_days: int
    auto_expand_resolve_window: bool
    max_series_points: int
    max_runtime_sec: int
    max_empty_pages: int
    panel_domain_quota: int
    panel_min_counts: Dict[int, int]
    min_active_domains: int
    min_per_domain: int
    max_domain_ratio: float
    max_pool_size: int
    min_final_count: int
    max_final_count: int
    target_pool_ratio: float
    selection_stability_patience: int
    domain_target_slots: int
    diversity_penalty: float
    strict_mode: bool
    materialize_batch_size: int
    output_dir: Path


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def to_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_dt(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.replace(" ", "T")
    # normalize timezone suffixes like +00 / -05 to +00:00 / -05:00
    m = re.search(r"([+-]\d{2})$", text)
    if m:
        text = text + ":00"
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def maybe_json_list(raw: Any) -> List[Any]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return []
        try:
            obj = json.loads(s)
            return obj if isinstance(obj, list) else []
        except json.JSONDecodeError:
            return []
    return []


def to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        try:
            return float(s)
        except ValueError:
            return None
    return None


def normalize_slug(text: str) -> str:
    t = text.strip().lower()
    t = re.sub(r"[^a-z0-9]+", "-", t)
    return re.sub(r"-+", "-", t).strip("-") or "other"


def duration_days(open_dt: datetime, close_dt: datetime) -> float:
    return max((close_dt - open_dt).total_seconds() / 86400.0, 0.0)


class PolymarketClient:
    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.trust_env = False
        self.session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": "future-forecasting-dataset-builder/3.0",
            }
        )

    def _request(
        self,
        method: str,
        base: str,
        path: str,
        endpoint_name: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json_body: Any = None,
        allow_404: bool = False,
    ) -> Tuple[Optional[Any], Optional[str]]:
        url = f"{base}{path}"
        for attempt in range(MAX_RETRIES + 1):
            try:
                time.sleep(RATE_LIMIT_SLEEP_SEC)
                resp = self.session.request(
                    method=method,
                    url=url,
                    params=params,
                    json=json_body,
                    timeout=REQUEST_TIMEOUT,
                )
            except requests.RequestException:
                if attempt == MAX_RETRIES:
                    return None, f"api_error:{endpoint_name}"
                time.sleep(BASE_BACKOFF_SEC * (2**attempt))
                continue

            if resp.status_code == 404 and allow_404:
                return None, None

            if resp.status_code in (429, 500, 502, 503, 504):
                if attempt == MAX_RETRIES:
                    return None, f"api_error:{endpoint_name}"
                if resp.status_code == 429:
                    ra = resp.headers.get("Retry-After")
                    wait = 1.0
                    if ra:
                        try:
                            wait = min(max(float(ra), 0.2), 3.0)
                        except ValueError:
                            wait = 1.0
                    time.sleep(wait)
                else:
                    time.sleep(BASE_BACKOFF_SEC * (2**attempt))
                continue

            if not (200 <= resp.status_code < 300):
                return None, f"api_error:{endpoint_name}"

            try:
                return resp.json(), None
            except ValueError:
                return None, f"api_error:{endpoint_name}"

        return None, f"api_error:{endpoint_name}"

    def list_markets_page(
        self, *, limit: int, offset: int, tag_id: Optional[str] = None
    ) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
        params = {
            "closed": "true",
            "limit": limit,
            "offset": offset,
            "order": "id",
            "ascending": "false",
            "include_tag": "true",
        }
        if tag_id is not None:
            params["tag_id"] = str(tag_id)
        data, err = self._request("GET", GAMMA_BASE, "/markets", "gamma_markets", params=params)
        if err:
            return None, err
        if not isinstance(data, list):
            return None, "api_error:gamma_markets"
        return data, None

    def get_tag_by_slug(self, slug: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        data, err = self._request(
            "GET",
            GAMMA_BASE,
            f"/tags/slug/{slug}",
            "gamma_tag_by_slug",
            allow_404=True,
        )
        if err:
            return None, err
        if data is None:
            return None, None
        if not isinstance(data, dict):
            return None, None
        return data, None

    def get_price_history(self, token_id: str) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
        params = {
            "market": token_id,
            "interval": "max",
            "fidelity": 1440,
        }
        data, err = self._request(
            "GET",
            CLOB_BASE,
            "/prices-history",
            "clob_prices_history",
            params=params,
            allow_404=True,
        )
        if err:
            return None, err
        if data is None:
            return [], None
        history = data.get("history") if isinstance(data, dict) else None
        if not isinstance(history, list):
            return [], None
        return history, None


def market_id(m: Dict[str, Any]) -> str:
    return str(m.get("id") or "")


def parse_outcomes(m: Dict[str, Any]) -> Tuple[List[str], List[float]]:
    outcomes = [str(x).strip() for x in maybe_json_list(m.get("outcomes"))]
    prices = [to_float(x) for x in maybe_json_list(m.get("outcomePrices"))]
    prices = [x for x in prices if x is not None]
    return outcomes, prices


def is_binary_outcomes(outcomes: List[str]) -> bool:
    return len(outcomes) == 2 and all(bool(x) for x in outcomes)


def resolved_status(m: Dict[str, Any], prices: List[float]) -> bool:
    if str(m.get("umaResolutionStatus") or "").strip().lower() == "resolved":
        return True
    if not bool(m.get("closed")) or len(prices) != 2:
        return False
    return max(prices) >= 0.95 and min(prices) <= 0.05


def extract_binary_answer(prices: List[float]) -> Optional[int]:
    if len(prices) != 2:
        return None
    if any(p is None for p in prices):
        return None
    return 1 if prices[0] >= prices[1] else 0


def primary_token_id(m: Dict[str, Any]) -> Optional[str]:
    ids = maybe_json_list(m.get("clobTokenIds"))
    if len(ids) < 1:
        return None
    return str(ids[0])


def parse_times(m: Dict[str, Any]) -> Tuple[Optional[datetime], Optional[datetime], Optional[datetime]]:
    open_dt = parse_dt(m.get("startDate") or m.get("createdAt"))
    close_dt = parse_dt(m.get("endDate") or m.get("closedTime"))
    resolve_dt = parse_dt(m.get("closedTime") or m.get("updatedAt"))
    return open_dt, close_dt, resolve_dt


def resolution_criteria(m: Dict[str, Any]) -> str:
    desc = str(m.get("description") or "").strip()
    source = str(m.get("resolutionSource") or "").strip()
    if source and source not in desc:
        return (desc + "\nResolution source: " + source).strip()
    return desc or source


def collect_official_tags(m: Dict[str, Any]) -> List[str]:
    vals: List[str] = []

    def add_tag_obj(t: Any) -> None:
        if isinstance(t, dict):
            for k in ("slug", "label"):
                v = t.get(k)
                if v:
                    vals.append(normalize_slug(str(v)))

    for t in (m.get("tags") or []):
        add_tag_obj(t)

    events = m.get("events") or []
    if isinstance(events, list):
        for e in events[:2]:
            if isinstance(e, dict):
                for t in (e.get("tags") or []):
                    add_tag_obj(t)
                cat = e.get("category")
                if cat:
                    vals.append(normalize_slug(str(cat)))

    cat = m.get("category")
    if cat:
        vals.append(normalize_slug(str(cat)))

    out: List[str] = []
    seen = set()
    for x in vals:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def infer_domain(m: Dict[str, Any]) -> str:
    tags = collect_official_tags(m)
    tagset = set(tags)

    if "sports" in tagset:
        return "sports"
    if "crypto" in tagset:
        return "finance"
    if tagset.intersection({"politics", "elections", "us-politics", "current-events", "global-elections"}):
        return "politics"
    if tagset.intersection({"equities", "stocks", "finance", "business", "macro"}):
        return "finance"
    if tagset.intersection({"weather", "temperature", "climate"}):
        return "weather"
    if tagset.intersection({"technology", "ai", "tech"}):
        return "tech"
    if tagset.intersection({"pop-culture", "entertainment", "movies", "music", "celebrity"}):
        return "culture"
    if tagset.intersection({"world", "geopolitics", "middle-east", "uk", "europe", "asia"}):
        return "world"

    for t in tags:
        if t not in GENERIC_TAGS:
            return t
    return "other"


def build_daily_series(history: List[Dict[str, Any]], resolve_dt: datetime, p_yes_final: float) -> List[Dict[str, Any]]:
    by_day: Dict[str, float] = {}
    for row in history:
        t = to_float(row.get("t"))
        p = to_float(row.get("p"))
        if t is None or p is None:
            continue
        if p < 0 or p > 1:
            continue
        day = datetime.fromtimestamp(int(t), tz=timezone.utc).date().isoformat()
        by_day[day] = p

    if not by_day:
        return [{"t": to_iso(resolve_dt), "p_yes": float(p_yes_final)}]

    return [
        {"t": to_iso(datetime.fromisoformat(day).replace(tzinfo=timezone.utc)), "p_yes": by_day[day]}
        for day in sorted(by_day)
    ]


def compress_series(series: List[Dict[str, Any]], max_points: int) -> List[Dict[str, Any]]:
    if len(series) <= max_points:
        return series

    # Keep the latest points for long markets instead of uniform downsampling.
    return series[-max_points:]


def series_span_days(series: List[Dict[str, Any]]) -> float:
    if len(series) < 2:
        return 0.0
    t0 = parse_dt(series[0].get("t"))
    t1 = parse_dt(series[-1].get("t"))
    if not t0 or not t1:
        return 0.0
    return max((t1 - t0).total_seconds() / 86400.0, 0.0)


def nearest_prob_at_or_before(
    series: List[Dict[str, Any]],
    target: datetime,
    *,
    max_forward_days: int = 0,
) -> Optional[float]:
    target_iso = to_iso(target)
    before = [x for x in series if x.get("t") and x["t"] <= target_iso]
    if before:
        return to_float(before[-1].get("p_yes"))
    if max_forward_days > 0:
        after = [x for x in series if x.get("t") and x["t"] > target_iso]
        if after:
            first = after[0]
            first_dt = parse_dt(first.get("t"))
            if first_dt is not None:
                lag_days = (first_dt - target).total_seconds() / 86400.0
                if lag_days <= float(max_forward_days):
                    return to_float(first.get("p_yes"))
    return None


def local_volatility(series: List[Dict[str, Any]], target: datetime, window_points: int = 8) -> float:
    target_iso = to_iso(target)
    vals = [to_float(x.get("p_yes")) for x in series if x.get("t") and x["t"] <= target_iso]
    vals = [v for v in vals if v is not None]
    vals = vals[-window_points:]
    if len(vals) < 2:
        return 0.0
    deltas = [abs(vals[i] - vals[i - 1]) for i in range(1, len(vals))]
    return sum(deltas) / float(len(deltas))


def difficulty_label(p_yes: Optional[float], vol: float) -> str:
    if p_yes is None:
        return "unknown"
    uncertainty = max(0.0, 1.0 - abs(p_yes - 0.5) * 2.0)  # near 0.5 => harder
    vol_norm = min(max(vol / 0.05, 0.0), 1.0)
    score = 0.6 * uncertainty + 0.4 * vol_norm
    if score >= 0.58:
        return "hard"
    if score >= 0.30:
        return "medium"
    return "easy"


def build_sampled_points(series: List[Dict[str, Any]], resolve_dt: datetime, offsets: List[int]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for d in offsets:
        key = f"{d}d"
        target = resolve_dt - timedelta(days=d)
        # Allow a small forward tolerance when exact/before point is missing.
        # This improves horizon coverage for sparse daily histories without using far-future points.
        forward_days = max(1, min(7, int(d // 4)))
        p = nearest_prob_at_or_before(series, target, max_forward_days=forward_days)
        if p is None:
            continue
        vol = local_volatility(series, target)
        out[key] = {
            "t": to_iso(target),
            "p_yes": p,
            "difficult": difficulty_label(p, vol),
        }
    return out


def coverage_score(item: Dict[str, Any]) -> Tuple[float, float]:
    return (float(item.get("_duration_days", 0.0)), float(item.get("_volume_total", 0.0)))


def balanced_sample_by_domain(records: List[Dict[str, Any]], sample_size: int) -> List[Dict[str, Any]]:
    if sample_size <= 0 or sample_size >= len(records):
        return list(records)

    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        groups[str(r.get("domain", "other"))].append(r)

    for arr in groups.values():
        arr.sort(key=lambda x: (-coverage_score(x)[0], -coverage_score(x)[1], str(x.get("market_id"))))

    keys = sorted(groups.keys())
    ptr = {k: 0 for k in keys}
    out: List[Dict[str, Any]] = []

    while len(out) < sample_size:
        progressed = False
        for k in keys:
            i = ptr[k]
            if i >= len(groups[k]):
                continue
            out.append(groups[k][i])
            ptr[k] += 1
            progressed = True
            if len(out) >= sample_size:
                break
        if not progressed:
            break
    return out


def dominant_difficulty(market: Dict[str, Any]) -> str:
    sampled = market.get("sampled", {})
    if not isinstance(sampled, dict):
        return "unknown"
    for k in ("30d", "7d", "1d"):
        if k in sampled and isinstance(sampled[k], dict):
            return str(sampled[k].get("difficult", "unknown"))
    vals = [str(v.get("difficult", "unknown")) for v in sampled.values() if isinstance(v, dict)]
    return vals[0] if vals else "unknown"


def balanced_sample_by_domain_and_difficulty(records: List[Dict[str, Any]], sample_size: int) -> List[Dict[str, Any]]:
    if sample_size <= 0 or sample_size >= len(records):
        return list(records)
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        key = (str(r.get("domain", "other")), dominant_difficulty(r))
        groups[key].append(r)
    for arr in groups.values():
        arr.sort(key=lambda x: str(x.get("market_id")))
    keys = sorted(groups.keys())
    ptr = {k: 0 for k in keys}
    out: List[Dict[str, Any]] = []
    while len(out) < sample_size:
        progressed = False
        for k in keys:
            i = ptr[k]
            if i >= len(groups[k]):
                continue
            out.append(groups[k][i])
            ptr[k] += 1
            progressed = True
            if len(out) >= sample_size:
                break
        if not progressed:
            break
    return out


def validate_schema(records: List[Dict[str, Any]], sample_offsets_days: List[int], n: int) -> None:
    if not records:
        print("schema_check: no records")
        return
    n = min(n, len(records))
    sample = random.sample(records, n)
    errors = 0

    required = [
        "market_id",
        "question",
        "domain",
        "description",
        "resolution_criteria",
        "open_time",
        "close_time",
        "resolve_time",
        "answer",
        "community_time_series",
        "sampled",
    ]
    allowed_sample_keys = {f"{d}d" for d in sample_offsets_days}

    for rec in sample:
        for k in required:
            if k not in rec:
                errors += 1

        if rec.get("answer") not in (0, 1):
            errors += 1

        prev_t = ""
        for row in rec.get("community_time_series", []):
            if set(row.keys()) != {"t", "p_yes"}:
                errors += 1
            p = to_float(row.get("p_yes"))
            if p is None or not (0 <= p <= 1):
                errors += 1
            t = row.get("t")
            if not isinstance(t, str):
                errors += 1
                continue
            if prev_t and t < prev_t:
                errors += 1
            prev_t = t

        s = rec.get("sampled", {})
        if not isinstance(s, dict) or not s:
            errors += 1
            continue
        for k, obj in s.items():
            if k not in allowed_sample_keys:
                continue
            if not isinstance(obj, dict):
                errors += 1
                continue
            if "p_yes" not in obj or "difficult" not in obj or "t" not in obj:
                errors += 1

    print(f"schema_check_samples={n} errors={errors}")


def horizon_validity_summary(records: List[Dict[str, Any]], sample_offsets_days: List[int]) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {f"{d}d": {"covered": 0, "missing": 0} for d in sample_offsets_days}
    for rec in records:
        resolve_dt = parse_dt(rec.get("resolve_time"))
        series = rec.get("community_time_series", [])
        first_dt = None
        if isinstance(series, list) and series:
            first_dt = parse_dt(series[0].get("t"))
        for d in sample_offsets_days:
            key = f"{d}d"
            sampled = (rec.get("sampled") or {}).get(key)
            p = to_float(sampled.get("p_yes")) if isinstance(sampled, dict) else None
            if p is None or resolve_dt is None or first_dt is None:
                out[key]["missing"] += 1
                continue
            out[key]["covered"] += 1
    return out


def panel_sort_key(m: Dict[str, Any]) -> Tuple[int, int, str]:
    sampled = m.get("sampled", {})
    sampled_count = len(sampled) if isinstance(sampled, dict) else 0
    series = m.get("community_time_series", [])
    series_len = len(series) if isinstance(series, list) else 0
    return (-sampled_count, -series_len, str(m.get("market_id", "")))


def build_panels(
    records: List[Dict[str, Any]],
    offsets: List[int],
    domain_quota: int,
    min_per_domain: int,
    max_domain_ratio: float,
) -> Dict[str, Dict[str, Any]]:
    panels: Dict[str, Dict[str, Any]] = {}
    for d in offsets:
        key = f"{d}d"
        eligible = [m for m in records if isinstance(m.get("sampled"), dict) and key in m["sampled"]]
        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for m in eligible:
            groups[str(m.get("domain", "other"))].append(m)
        for arr in groups.values():
            arr.sort(key=panel_sort_key)

        # Soft balance:
        # 1) start from all eligible records (no pre-cap) to avoid artificial exact-equal counts
        # 2) trim dominant domains only if ratio gets too high
        selected_by_domain: Dict[str, List[Dict[str, Any]]] = {}
        for domain in sorted(groups.keys()):
            selected_by_domain[domain] = list(groups[domain])

        def counts_map() -> Dict[str, int]:
            return {dname: len(arr) for dname, arr in selected_by_domain.items() if arr}

        domain_counts = counts_map()
        # Trim only when imbalance is too high among active domains.
        while True:
            active = {dname: c for dname, c in domain_counts.items() if c >= min_per_domain}
            if len(active) < 2:
                break
            mn = min(active.values())
            mx = max(active.values())
            if mn <= 0 or (mx / mn) <= max_domain_ratio:
                break
            biggest = max(active.items(), key=lambda kv: (kv[1], kv[0]))[0]
            # Keep a loose floor near quota so trimming does not erase domain coverage.
            if len(selected_by_domain.get(biggest, [])) <= max(domain_quota, min_per_domain):
                break
            if not selected_by_domain.get(biggest):
                break
            selected_by_domain[biggest].pop()
            domain_counts = counts_map()

        selected: List[Dict[str, Any]] = []
        for domain in sorted(selected_by_domain.keys()):
            selected.extend(selected_by_domain[domain])
        domain_counts = counts_map()

        selected.sort(key=lambda x: str(x.get("market_id", "")))
        panels[key] = {
            "offset_days": d,
            "domain_quota": domain_quota,
            "count": len(selected),
            "domain_counts": dict(sorted(domain_counts.items())),
            "markets": selected,
        }
    return panels


def panel_goal_status(
    panels: Dict[str, Dict[str, Any]],
    offsets: List[int],
    panel_min_counts: Dict[int, int],
    min_active_domains: int,
    min_per_domain: int,
    max_domain_ratio: float,
) -> Tuple[bool, Dict[str, Dict[str, Any]]]:
    status: Dict[str, Dict[str, Any]] = {}
    all_ok = True
    for d in offsets:
        key = f"{d}d"
        panel = panels.get(key, {})
        domain_counts = panel.get("domain_counts", {}) if isinstance(panel, dict) else {}
        if not isinstance(domain_counts, dict):
            domain_counts = {}
        min_count = int(panel_min_counts.get(d, 0))
        total = int(panel.get("count", 0)) if isinstance(panel, dict) else 0
        active_vals = [int(v) for v in domain_counts.values() if int(v) >= min_per_domain]
        active_domains = len(active_vals)
        ratio = float("inf")
        if active_vals:
            ratio = max(active_vals) / max(1, min(active_vals))

        count_ok = total >= min_count
        active_ok = active_domains >= min_active_domains
        ratio_ok = ratio <= max_domain_ratio if active_vals else False
        ok = count_ok and active_ok and ratio_ok
        if not ok:
            all_ok = False

        status[key] = {
            "count": total,
            "min_count": min_count,
            "count_ok": count_ok,
            "active_domains": active_domains,
            "min_active_domains": min_active_domains,
            "active_ok": active_ok,
            "domain_ratio": None if ratio == float("inf") else round(ratio, 4),
            "max_domain_ratio": max_domain_ratio,
            "ratio_ok": ratio_ok,
            "ok": ok,
        }
    return all_ok, status


def final_count_from_panels(panels: Dict[str, Dict[str, Any]]) -> int:
    ids = set()
    for panel in panels.values():
        if not isinstance(panel, dict):
            continue
        for m in panel.get("markets", []):
            if isinstance(m, dict):
                ids.add(str(m.get("market_id", "")))
    return len(ids)


def panel_domain_counts(records: List[Dict[str, Any]], offsets: List[int]) -> Dict[str, Counter]:
    counts: Dict[str, Counter] = {f"{d}d": Counter() for d in offsets}
    for m in records:
        dom = str(m.get("domain", "other"))
        sampled = m.get("sampled", {})
        if not isinstance(sampled, dict):
            continue
        for d in offsets:
            k = f"{d}d"
            if k in sampled:
                counts[k][dom] += 1
    return counts


def candidate_deficit_score(
    row: Dict[str, Any],
    counts: Dict[str, Counter],
    offsets: List[int],
    quota: int,
) -> float:
    dom = str(row.get("domain", "other"))
    dur = float(row.get("_duration_days", 0.0))
    horizon_weight = {30: 3.0, 7: 1.4, 1: 1.0}
    score = 0.0
    for d in offsets:
        if dur < d:
            continue
        k = f"{d}d"
        cur = counts.get(k, Counter()).get(dom, 0)
        gap = max(quota - cur, 0)
        score += float(gap) * float(horizon_weight.get(d, 1.0))
    return score


def normalized_question_key(text: str) -> str:
    key = re.sub(r"[^a-z0-9]+", " ", str(text or "").strip().lower())
    key = re.sub(r"\s+", " ", key).strip()
    return key


def horizon_weights(offsets: List[int]) -> Dict[int, float]:
    wanted = [30, 7, 1]
    if sorted(offsets) == sorted(wanted):
        return {30: 1.0 / 3.0, 7: 1.0 / 3.0, 1: 1.0 / 3.0}
    # fallback for custom offsets: shorter horizon gets larger weight
    raw = {d: 1.0 / max(float(d), 1.0) ** 0.5 for d in offsets}
    denom = sum(raw.values()) or 1.0
    return {d: raw[d] / denom for d in offsets}


def derive_target_count(pool_size: int, *, min_final: int, max_final: int, ratio: float) -> int:
    if pool_size <= 0:
        return 0
    ratio_target = int(round(float(pool_size) * float(ratio)))
    target = max(min_final, ratio_target)
    target = min(target, max_final)
    return max(1, min(target, pool_size))


def derive_horizon_targets(
    offsets: List[int],
    *,
    min_counts: Dict[int, int],
    total_target_count: int,
) -> Dict[str, int]:
    w = horizon_weights(offsets)
    out: Dict[str, int] = {}
    for d in offsets:
        k = f"{d}d"
        base = int(round(float(total_target_count) * float(w.get(d, 0.0))))
        out[k] = max(int(min_counts.get(d, 0)), base)
    return out


def pool_availability(
    pool: List[Dict[str, Any]],
    offsets: List[int],
) -> Tuple[Dict[str, Counter], Dict[str, Counter]]:
    domains: Dict[str, Counter] = {f"{d}d": Counter() for d in offsets}
    diffs: Dict[str, Counter] = {f"{d}d": Counter() for d in offsets}
    for row in pool:
        dom = str(row.get("domain", "other"))
        sampled = row.get("sampled", {})
        if not isinstance(sampled, dict):
            continue
        for d in offsets:
            k = f"{d}d"
            obj = sampled.get(k)
            if not isinstance(obj, dict):
                continue
            domains[k][dom] += 1
            diff = str(obj.get("difficult", "unknown"))
            diffs[k][diff] += 1
    return domains, diffs


def derive_effective_selection_policy(
    *,
    cfg: Config,
    offsets: List[int],
    pool_size: int,
    pool_horizon_counts: Dict[str, int],
    domain_avail: Dict[str, Counter],
) -> Dict[str, Any]:
    supply_ratio = min(float(pool_size) / float(max(int(cfg.min_final_count), 1)), 1.0)
    shortage_ratio = max(0.0, 1.0 - supply_ratio)
    if shortage_ratio <= 0.08:
        relax_level = "strict"
    elif shortage_ratio <= 0.25:
        relax_level = "mild"
    elif shortage_ratio <= 0.45:
        relax_level = "moderate"
    else:
        relax_level = "aggressive"

    min_per_domain = int(cfg.min_per_domain)
    if shortage_ratio > 0.0:
        min_per_domain = max(3, int(round(float(cfg.min_per_domain) * (1.0 - 0.55 * shortage_ratio))))

    max_domain_ratio = float(cfg.max_domain_ratio)
    if shortage_ratio > 0.0:
        max_domain_ratio = min(
            float(cfg.max_domain_ratio) * (1.0 + 1.2 * shortage_ratio),
            max(float(cfg.max_domain_ratio) + 4.0, float(cfg.max_domain_ratio) * 2.5),
        )

    panel_min_counts: Dict[int, int] = {}
    for d in offsets:
        hk = f"{d}d"
        required = int(cfg.panel_min_counts.get(d, 0))
        available = int(pool_horizon_counts.get(hk, 0))
        effective = min(required, available)
        if shortage_ratio > 0.0:
            effective = int(round(float(effective) * (1.0 - 0.35 * shortage_ratio)))
        if required > 0 and available > 0 and effective <= 0:
            effective = 1
        panel_min_counts[int(d)] = max(0, int(effective))

    feasible_domains: List[int] = []
    for d in offsets:
        hk = f"{d}d"
        ctr = domain_avail.get(hk, Counter())
        feasible = sum(1 for _dom, cnt in ctr.items() if int(cnt) >= int(min_per_domain))
        if feasible <= 0:
            feasible = len(ctr)
        if feasible > 0:
            feasible_domains.append(int(feasible))
    min_active_domains = int(cfg.min_active_domains)
    if feasible_domains:
        min_active_domains = max(1, min(int(cfg.min_active_domains), min(feasible_domains)))

    domain_target_slots = int(cfg.domain_target_slots)
    if shortage_ratio > 0.25:
        domain_target_slots = max(
            domain_target_slots,
            int(cfg.domain_target_slots + round(2.0 + 6.0 * shortage_ratio)),
        )

    return {
        "relax_level": relax_level,
        "supply_ratio": round(supply_ratio, 4),
        "shortage_ratio": round(shortage_ratio, 4),
        "panel_min_counts": dict(sorted(panel_min_counts.items())),
        "min_active_domains": int(min_active_domains),
        "min_per_domain": int(min_per_domain),
        "max_domain_ratio": float(max_domain_ratio),
        "domain_target_slots": int(domain_target_slots),
    }


def derive_domain_targets(
    domain_avail: Dict[str, Counter],
    horizon_targets: Dict[str, int],
    *,
    min_active_domains: int,
    min_per_domain: int,
    domain_target_slots: int,
) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for hk, ctr in domain_avail.items():
        doms = sorted(ctr.items(), key=lambda kv: (-int(kv[1]), kv[0]))
        if not doms:
            out[hk] = {}
            continue
        slots = max(min_active_domains, min(domain_target_slots, len(doms)))
        chosen = [name for name, _c in doms[:slots]]
        per = max(
            min_per_domain,
            int((int(horizon_targets.get(hk, 0)) + max(len(chosen), 1) - 1) / max(len(chosen), 1)),
        )
        out[hk] = {name: per for name in chosen}
    return out


def derive_difficulty_targets(
    diff_avail: Dict[str, Counter],
    horizon_targets: Dict[str, int],
) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for hk, ctr in diff_avail.items():
        classes = [d for d in ("easy", "medium", "hard") if int(ctr.get(d, 0)) > 0]
        if not classes:
            out[hk] = {}
            continue
        per = int((int(horizon_targets.get(hk, 0)) + len(classes) - 1) / len(classes))
        out[hk] = {d: per for d in classes}
    return out


def derive_global_domain_targets(
    pool_domain_totals: Counter,
    *,
    target_count: int,
    min_per_domain: int,
    domain_target_slots: int,
) -> Dict[str, int]:
    if not pool_domain_totals or target_count <= 0:
        return {}

    ranked = sorted(pool_domain_totals.items(), key=lambda kv: (-int(kv[1]), kv[0]))
    eligible = [(d, c) for d, c in ranked if int(c) >= int(min_per_domain)]
    if not eligible:
        eligible = ranked
    chosen = eligible[: max(1, int(domain_target_slots))]
    capacities = {d: int(c) for d, c in chosen}
    targets = {d: 0 for d in capacities.keys()}

    remaining_target = int(target_count)
    active = set(capacities.keys())
    while remaining_target > 0 and active:
        per = max(1, int((remaining_target + len(active) - 1) / len(active)))
        progressed = False
        to_remove: List[str] = []
        for d in sorted(active):
            cap = int(capacities[d])
            cur = int(targets[d])
            room = max(cap - cur, 0)
            if room <= 0:
                to_remove.append(d)
                continue
            add = min(per, room, remaining_target)
            if add > 0:
                targets[d] = cur + add
                remaining_target -= add
                progressed = True
            if targets[d] >= cap:
                to_remove.append(d)
            if remaining_target <= 0:
                break
        for d in to_remove:
            active.discard(d)
        if not progressed:
            break
    return dict(sorted(targets.items()))


def evaluate_hard_constraints(
    *,
    offsets: List[int],
    horizon_counts: Counter,
    domain_counts: Dict[str, Counter],
    panel_min_counts: Dict[int, int],
    min_active_domains: int,
    min_per_domain: int,
    max_domain_ratio: float,
) -> Dict[str, Any]:
    status: Dict[str, Any] = {}
    all_ok = True
    for d in offsets:
        hk = f"{d}d"
        ctr = domain_counts.get(hk, Counter())
        total = int(horizon_counts.get(hk, 0))
        min_count = int(panel_min_counts.get(d, 0))
        count_gap = max(min_count - total, 0)
        active_vals = [int(v) for v in ctr.values() if int(v) >= min_per_domain]
        active_domains = len(active_vals)
        active_gap = max(min_active_domains - active_domains, 0)
        ratio = None
        ratio_ok = True
        if len(active_vals) >= 2:
            mn = min(active_vals)
            mx = max(active_vals)
            ratio = float(mx) / float(max(1, mn))
            ratio_ok = ratio <= max_domain_ratio
        ok = (count_gap == 0) and (active_gap == 0) and ratio_ok
        if not ok:
            all_ok = False
        status[hk] = {
            "count": total,
            "min_count": min_count,
            "count_gap": count_gap,
            "active_domains": active_domains,
            "min_active_domains": min_active_domains,
            "active_gap": active_gap,
            "domain_ratio": None if ratio is None else round(ratio, 4),
            "max_domain_ratio": max_domain_ratio,
            "ratio_ok": ratio_ok,
            "ok": ok,
            "domain_counts": dict(sorted(ctr.items())),
        }
    return {"all_ok": all_ok, "per_horizon": status}


def market_marginal_score(
    row: Dict[str, Any],
    *,
    offsets: List[int],
    horizon_counts: Counter,
    domain_counts: Dict[str, Counter],
    diff_counts: Dict[str, Counter],
    global_domain_counts: Counter,
    horizon_targets: Dict[str, int],
    domain_targets: Dict[str, Dict[str, int]],
    diff_targets: Dict[str, Dict[str, int]],
    global_domain_targets: Dict[str, int],
    question_key_counts: Counter,
    diversity_penalty: float,
) -> float:
    sampled = row.get("sampled", {})
    if not isinstance(sampled, dict):
        return -1e9
    dom = str(row.get("domain", "other"))
    score = 0.0
    for d in offsets:
        hk = f"{d}d"
        obj = sampled.get(hk)
        if not isinstance(obj, dict):
            continue

        # Global horizon coverage gain.
        cur_h = int(horizon_counts.get(hk, 0))
        tgt_h = int(horizon_targets.get(hk, 0))
        score += 1.4 * float(max(tgt_h - cur_h, 0))

        # Domain balance gain under each horizon.
        cur_hd = int(domain_counts.get(hk, Counter()).get(dom, 0))
        tgt_hd = int(domain_targets.get(hk, {}).get(dom, 0))
        if tgt_hd > 0:
            score += 2.0 * float(max(tgt_hd - cur_hd, 0))
            score -= 0.9 * float(max(cur_hd - tgt_hd, 0))
        else:
            score -= 0.8

        # Difficulty balance gain under each horizon.
        diff = str(obj.get("difficult", "unknown"))
        tgt_diff = int(diff_targets.get(hk, {}).get(diff, 0))
        if tgt_diff > 0:
            cur_diff = int(diff_counts.get(hk, Counter()).get(diff, 0))
            score += 1.1 * float(max(tgt_diff - cur_diff, 0))
            score -= 0.4 * float(max(cur_diff - tgt_diff, 0))

    # Global domain balance gain/penalty.
    tgt_dom = int(global_domain_targets.get(dom, 0))
    cur_dom = int(global_domain_counts.get(dom, 0))
    if tgt_dom > 0:
        score += 2.4 * float(max(tgt_dom - cur_dom, 0))
        score -= 1.8 * float(max(cur_dom - tgt_dom, 0))
    else:
        score -= 1.2

    # Quality tie-breakers.
    score += min(float(row.get("_volume_total", 0.0)) / 12000.0, 0.8)
    score += min(float(row.get("_duration_days", 0.0)) / 90.0, 0.6)

    # Deduplicate near-identical questions.
    qk = normalized_question_key(str(row.get("question", "")))
    if qk and int(question_key_counts.get(qk, 0)) > 0:
        score -= float(diversity_penalty)
    return score


def global_select_markets(
    pool: List[Dict[str, Any]],
    cfg: Config,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not pool:
        return [], {
            "selected_count": 0,
            "pool_count": 0,
            "pool_horizon_counts": {f"{d}d": 0 for d in cfg.sample_offsets_days},
            "target_count": 0,
            "hard_constraints_met": False,
            "hard_constraint_status": {},
            "horizon_targets": {},
            "domain_targets": {},
            "difficulty_targets": {},
            "difficulty_counts": {},
            "difficulty_gaps": {},
            "stopped_by_stability": False,
            "impossible_hard_constraints": False,
            "impossible_hard_horizons": {},
            "global_domain_targets": {},
            "global_domain_counts": {},
            "global_domain_gaps": {},
            "impossible_domain_constraints": False,
            "impossible_domain_gaps": {},
            "configured_constraints": {
                "panel_min_counts": dict(sorted((int(k), int(v)) for k, v in cfg.panel_min_counts.items())),
                "min_active_domains": int(cfg.min_active_domains),
                "min_per_domain": int(cfg.min_per_domain),
                "max_domain_ratio": float(cfg.max_domain_ratio),
                "domain_target_slots": int(cfg.domain_target_slots),
            },
            "effective_policy": {},
        }

    offsets = list(cfg.sample_offsets_days)
    domain_avail, diff_avail = pool_availability(pool, offsets)
    pool_domain_totals = Counter(str(row.get("domain", "other")) for row in pool)
    pool_horizon_counts = {
        f"{d}d": int(sum(domain_avail.get(f"{d}d", Counter()).values()))
        for d in offsets
    }
    effective_policy = derive_effective_selection_policy(
        cfg=cfg,
        offsets=offsets,
        pool_size=len(pool),
        pool_horizon_counts=pool_horizon_counts,
        domain_avail=domain_avail,
    )
    effective_panel_min_counts = {
        int(k): int(v)
        for k, v in dict(effective_policy.get("panel_min_counts", {})).items()
    }
    effective_min_active_domains = int(effective_policy.get("min_active_domains", cfg.min_active_domains))
    effective_min_per_domain = int(effective_policy.get("min_per_domain", cfg.min_per_domain))
    effective_max_domain_ratio = float(effective_policy.get("max_domain_ratio", cfg.max_domain_ratio))
    effective_domain_target_slots = int(effective_policy.get("domain_target_slots", cfg.domain_target_slots))

    target_count = derive_target_count(
        len(pool),
        min_final=cfg.min_final_count,
        max_final=cfg.max_final_count,
        ratio=cfg.target_pool_ratio,
    )
    horizon_targets = derive_horizon_targets(
        offsets,
        min_counts=effective_panel_min_counts,
        total_target_count=target_count,
    )
    domain_targets = derive_domain_targets(
        domain_avail,
        horizon_targets,
        min_active_domains=effective_min_active_domains,
        min_per_domain=effective_min_per_domain,
        domain_target_slots=effective_domain_target_slots,
    )
    global_domain_targets = derive_global_domain_targets(
        pool_domain_totals,
        target_count=target_count,
        min_per_domain=effective_min_per_domain,
        domain_target_slots=effective_domain_target_slots,
    )
    diff_targets = derive_difficulty_targets(diff_avail, horizon_targets)

    selected_idx: List[int] = []
    remaining: Set[int] = set(range(len(pool)))
    remaining_domain_counts: Counter = Counter(pool_domain_totals)
    remaining_horizon_counts: Counter = Counter()
    for i, row in enumerate(pool):
        s = row.get("sampled", {})
        if not isinstance(s, dict):
            continue
        for d in offsets:
            hk = f"{d}d"
            if isinstance(s.get(hk), dict):
                remaining_horizon_counts[hk] += 1
    horizon_counts: Counter = Counter()
    domain_counts: Dict[str, Counter] = {f"{d}d": Counter() for d in offsets}
    diff_counts: Dict[str, Counter] = {f"{d}d": Counter() for d in offsets}
    global_domain_counts: Counter = Counter()
    question_key_counts: Counter = Counter()

    no_gain_streak = 0
    repair_cap = min(len(pool), max(int(cfg.max_final_count), int(target_count)) + 250)
    impossible_hard = False
    impossible_hard_horizons: Dict[str, Dict[str, int]] = {}
    impossible_domain = False
    impossible_domain_gaps: Dict[str, Dict[str, int]] = {}
    while remaining and len(selected_idx) < repair_cap:
        hard_now = evaluate_hard_constraints(
            offsets=offsets,
            horizon_counts=horizon_counts,
            domain_counts=domain_counts,
            panel_min_counts=effective_panel_min_counts,
            min_active_domains=effective_min_active_domains,
            min_per_domain=effective_min_per_domain,
            max_domain_ratio=effective_max_domain_ratio,
        )
        hard_met = bool(hard_now.get("all_ok"))
        reached_target = len(selected_idx) >= int(target_count)
        if reached_target and no_gain_streak >= int(cfg.selection_stability_patience):
            break

        impossible_hard_horizons = {}
        for d in offsets:
            hk = f"{d}d"
            req = int(effective_panel_min_counts.get(d, 0))
            cur = int(horizon_counts.get(hk, 0))
            remain = int(remaining_horizon_counts.get(hk, 0))
            if cur + remain < req:
                impossible_hard_horizons[hk] = {
                    "required": req,
                    "current": cur,
                    "remaining_potential": remain,
                    "max_possible": cur + remain,
                    "gap_unreachable": req - (cur + remain),
                }
        impossible_hard = bool(impossible_hard_horizons)
        impossible_domain_gaps = {}
        for dom, req in sorted(global_domain_targets.items()):
            cur = int(global_domain_counts.get(dom, 0))
            remain = int(remaining_domain_counts.get(dom, 0))
            if cur + remain < int(req):
                impossible_domain_gaps[dom] = {
                    "required": int(req),
                    "current": cur,
                    "remaining_potential": remain,
                    "max_possible": cur + remain,
                    "gap_unreachable": int(req) - (cur + remain),
                }
        impossible_domain = bool(impossible_domain_gaps)

        if reached_target and (impossible_hard or impossible_domain) and no_gain_streak >= 2:
            break

        best_i = None
        best_score = -1e18
        for i in remaining:
            score = market_marginal_score(
                pool[i],
                offsets=offsets,
                horizon_counts=horizon_counts,
                domain_counts=domain_counts,
                diff_counts=diff_counts,
                global_domain_counts=global_domain_counts,
                horizon_targets=horizon_targets,
                domain_targets=domain_targets,
                diff_targets=diff_targets,
                global_domain_targets=global_domain_targets,
                question_key_counts=question_key_counts,
                diversity_penalty=cfg.diversity_penalty,
            )
            if score > best_score:
                best_i = i
                best_score = score

        if best_i is None:
            break

        if reached_target and best_score <= 0:
            no_gain_streak += 1
        else:
            no_gain_streak = 0

        if reached_target and no_gain_streak >= int(cfg.selection_stability_patience):
            break

        row = pool[best_i]
        sampled = row.get("sampled", {})
        dom = str(row.get("domain", "other"))
        for d in offsets:
            hk = f"{d}d"
            obj = sampled.get(hk) if isinstance(sampled, dict) else None
            if not isinstance(obj, dict):
                continue
            horizon_counts[hk] += 1
            domain_counts[hk][dom] += 1
            diff = str(obj.get("difficult", "unknown"))
            diff_counts[hk][diff] += 1
        qk = normalized_question_key(str(row.get("question", "")))
        if qk:
            question_key_counts[qk] += 1
        global_domain_counts[dom] += 1

        selected_idx.append(best_i)
        remaining.remove(best_i)
        remaining_domain_counts[dom] = max(0, int(remaining_domain_counts.get(dom, 0)) - 1)
        s_sel = row.get("sampled", {})
        if isinstance(s_sel, dict):
            for d in offsets:
                hk = f"{d}d"
                if isinstance(s_sel.get(hk), dict):
                    remaining_horizon_counts[hk] = max(0, int(remaining_horizon_counts.get(hk, 0)) - 1)

        # Soft cap for final output size if constraints already good and no gain.
        if len(selected_idx) >= cfg.max_final_count and no_gain_streak > 0:
            break

    selected = [pool[i] for i in selected_idx]
    selected = sorted(selected, key=lambda x: str(x.get("market_id", "")))

    final_hard = evaluate_hard_constraints(
        offsets=offsets,
        horizon_counts=horizon_counts,
        domain_counts=domain_counts,
        panel_min_counts=effective_panel_min_counts,
        min_active_domains=effective_min_active_domains,
        min_per_domain=effective_min_per_domain,
        max_domain_ratio=effective_max_domain_ratio,
    )

    difficulty_gaps: Dict[str, Dict[str, int]] = {}
    for hk, tgt in diff_targets.items():
        gap_map: Dict[str, int] = {}
        for cls, tv in tgt.items():
            gap_map[cls] = max(int(tv) - int(diff_counts.get(hk, Counter()).get(cls, 0)), 0)
        difficulty_gaps[hk] = gap_map

    global_domain_gaps: Dict[str, int] = {}
    for dom, req in sorted(global_domain_targets.items()):
        global_domain_gaps[dom] = max(int(req) - int(global_domain_counts.get(dom, 0)), 0)

    report = {
        "selected_count": len(selected),
        "pool_count": len(pool),
        "pool_horizon_counts": pool_horizon_counts,
        "target_count": int(target_count),
        "hard_constraints_met": bool(final_hard.get("all_ok", False)),
        "hard_constraint_status": final_hard.get("per_horizon", {}),
        "global_domain_targets": global_domain_targets,
        "global_domain_counts": dict(sorted(global_domain_counts.items())),
        "global_domain_gaps": global_domain_gaps,
        "horizon_targets": horizon_targets,
        "domain_targets": {
            hk: dict(sorted(v.items()))
            for hk, v in sorted(domain_targets.items())
        },
        "difficulty_targets": {
            hk: dict(sorted(v.items()))
            for hk, v in sorted(diff_targets.items())
        },
        "difficulty_counts": {
            hk: dict(sorted(diff_counts.get(hk, Counter()).items()))
            for hk in sorted(diff_counts.keys())
        },
        "difficulty_gaps": difficulty_gaps,
        "stopped_by_stability": bool(no_gain_streak >= int(cfg.selection_stability_patience)),
        "impossible_hard_constraints": impossible_hard,
        "impossible_hard_horizons": impossible_hard_horizons,
        "impossible_domain_constraints": impossible_domain,
        "impossible_domain_gaps": impossible_domain_gaps,
        "configured_constraints": {
            "panel_min_counts": dict(sorted((int(k), int(v)) for k, v in cfg.panel_min_counts.items())),
            "min_active_domains": int(cfg.min_active_domains),
            "min_per_domain": int(cfg.min_per_domain),
            "max_domain_ratio": float(cfg.max_domain_ratio),
            "domain_target_slots": int(cfg.domain_target_slots),
        },
        "effective_policy": {
            "relax_level": str(effective_policy.get("relax_level", "strict")),
            "supply_ratio": float(effective_policy.get("supply_ratio", 1.0)),
            "shortage_ratio": float(effective_policy.get("shortage_ratio", 0.0)),
            "panel_min_counts": dict(
                sorted((int(k), int(v)) for k, v in effective_panel_min_counts.items())
            ),
            "min_active_domains": int(effective_min_active_domains),
            "min_per_domain": int(effective_min_per_domain),
            "max_domain_ratio": float(effective_max_domain_ratio),
            "domain_target_slots": int(effective_domain_target_slots),
        },
    }
    return selected, report


def build_dataset(cfg: Config) -> Tuple[Dict[str, Any], Counter]:
    client = PolymarketClient()
    reject_ctr: Counter = Counter()

    candidates: List[Dict[str, Any]] = []
    candidate_ids = set()
    terminal_rejected_ids = set()
    seen_ids = set()
    scanned_ids = set()
    tried_history_ids = set()
    kept_market_ids = set()
    domain_sources: List[Tuple[str, str]] = []
    for domain, slugs in DOMAIN_TAG_SLUGS.items():
        found_id: Optional[str] = None
        for slug in slugs:
            tag_obj, tag_err = client.get_tag_by_slug(slug)
            if tag_err:
                reject_ctr[tag_err] += 1
                continue
            if not tag_obj:
                continue
            tid = tag_obj.get("id")
            if tid is not None:
                found_id = str(tid)
                break
        if found_id is not None:
            domain_sources.append((domain, found_id))

    if not domain_sources:
        domain_sources = [("global", "")]

    progress = None
    scan_progress = None
    if tqdm is not None:
        progress = tqdm(desc="Building pool markets", unit="mkt")
        scan_progress = tqdm(total=0, desc="Scanning markets", unit="mkt")

    markets: List[Dict[str, Any]] = []
    live_horizon_counts: Counter = Counter({f"{d}d": 0 for d in cfg.sample_offsets_days})
    offset = 0
    empty_page_streak = 0
    current_recent_resolve_days = cfg.recent_resolve_days
    recent_cutoff = utc_now() - timedelta(days=current_recent_resolve_days)
    recent_window_expansions = 0
    runtime_extended = False
    runtime_soft_cap_sec = int(cfg.max_runtime_sec)
    runtime_hard_cap_sec = int(max(cfg.max_runtime_sec + 60, cfg.max_runtime_sec * 5 // 2))
    start_monotonic = time.monotonic()
    stop_reason = "exhausted"

    # Build with sequential streaming pagination until one stop condition is met.
    while True:
        elapsed = time.monotonic() - start_monotonic
        stop_trigger: Optional[str] = None
        if elapsed >= runtime_soft_cap_sec:
            need_more_pool = len(markets) < cfg.min_final_count
            can_extend_runtime = need_more_pool and elapsed < runtime_hard_cap_sec and empty_page_streak < cfg.max_empty_pages
            if can_extend_runtime:
                runtime_extended = True
            else:
                stop_trigger = "max_runtime_reached"
        elif empty_page_streak >= cfg.max_empty_pages:
            stop_trigger = "max_empty_pages_reached"
        elif len(markets) >= cfg.max_pool_size:
            stop_trigger = "max_pool_size_reached"

        if stop_trigger:
            panels_now = build_panels(
                markets,
                cfg.sample_offsets_days,
                cfg.panel_domain_quota,
                cfg.min_per_domain,
                cfg.max_domain_ratio,
            )
            goals_met_now, _ = panel_goal_status(
                panels_now,
                cfg.sample_offsets_days,
                cfg.panel_min_counts,
                cfg.min_active_domains,
                cfg.min_per_domain,
                cfg.max_domain_ratio,
            )
            final_count_now = final_count_from_panels(panels_now)
            need_more = (not goals_met_now) or (final_count_now < cfg.min_final_count)
            can_expand_window = (
                cfg.auto_expand_resolve_window
                and stop_trigger != "max_runtime_reached"
                and need_more
                and current_recent_resolve_days < cfg.max_recent_resolve_days
            )
            if can_expand_window:
                old_days = current_recent_resolve_days
                current_recent_resolve_days = min(
                    cfg.max_recent_resolve_days,
                    current_recent_resolve_days + cfg.expand_resolve_step_days,
                )
                recent_cutoff = utc_now() - timedelta(days=current_recent_resolve_days)
                recent_window_expansions += 1
                offset = 0
                empty_page_streak = 0
                seen_ids.clear()
                stop_reason = f"expanded_recent_window:{old_days}->{current_recent_resolve_days}"
                continue
            stop_reason = stop_trigger
            break

        before_candidates = len(candidates)
        before_markets = len(markets)

        for _src_domain, src_tag_id in domain_sources:
            if len(markets) >= cfg.max_pool_size:
                break
            page, err = client.list_markets_page(limit=cfg.page_limit, offset=offset, tag_id=(src_tag_id or None))
            if err:
                reject_ctr[err] += 1
                continue
            if not page:
                continue

            for m in page:
                mid = market_id(m)
                if (
                    not mid
                    or mid in seen_ids
                    or mid in candidate_ids
                    or mid in terminal_rejected_ids
                    or mid in tried_history_ids
                    or mid in kept_market_ids
                ):
                    continue
                seen_ids.add(mid)
                if mid not in scanned_ids:
                    scanned_ids.add(mid)
                    if scan_progress is not None:
                        scan_progress.update(1)

                outcomes, prices = parse_outcomes(m)
                if not resolved_status(m, prices):
                    reject_ctr["unresolved"] += 1
                    terminal_rejected_ids.add(mid)
                    continue
                if not is_binary_outcomes(outcomes):
                    reject_ctr["non_binary_outcomes"] += 1
                    terminal_rejected_ids.add(mid)
                    continue

                open_dt, close_dt, resolve_dt = parse_times(m)
                if open_dt is None or close_dt is None or resolve_dt is None:
                    reject_ctr["missing_resolution"] += 1
                    terminal_rejected_ids.add(mid)
                    continue
                if resolve_dt < recent_cutoff:
                    reject_ctr["old_resolve_time"] += 1
                    continue

                if cfg.min_duration_days > 0 and duration_days(open_dt, close_dt) < float(cfg.min_duration_days):
                    reject_ctr["short_duration"] += 1
                    terminal_rejected_ids.add(mid)
                    continue

                crit = resolution_criteria(m)

                volume_total = to_float(m.get("volumeNum"))
                if volume_total is None:
                    volume_total = to_float(m.get("volume"))

                if (volume_total or 0.0) < cfg.min_volume_total:
                    reject_ctr["low_volume"] += 1
                    terminal_rejected_ids.add(mid)
                    continue

                ans = extract_binary_answer(prices)
                if ans is None:
                    reject_ctr["missing_resolution"] += 1
                    terminal_rejected_ids.add(mid)
                    continue

                p_yes_final = float(prices[0])
                yid = primary_token_id(m)
                domain = infer_domain(m)

                candidates.append(
                    {
                        "market_id": mid,
                        "question": str(m.get("question") or "").strip(),
                        "domain": domain,
                        "description": str(m.get("description") or "").strip() or None,
                        "resolution_criteria": crit or "",
                        "open_time": to_iso(open_dt),
                        "close_time": to_iso(close_dt),
                        "resolve_time": to_iso(resolve_dt),
                        "answer": ans,
                        "community_time_series": [],
                        "sampled": {},
                        "_duration_days": duration_days(open_dt, close_dt),
                        "_volume_total": float(volume_total or 0.0),
                        "_yes_token_id": yid,
                        "_p_yes_final": p_yes_final,
                    }
                )
                candidate_ids.add(mid)

        # materialize incrementally after each sequential offset batch
        remaining = [c for c in candidates if c["market_id"] not in tried_history_ids]
        if remaining:
            current_counts = panel_domain_counts(markets, cfg.sample_offsets_days)
            scored_remaining = [
                (
                    candidate_deficit_score(x, current_counts, cfg.sample_offsets_days, cfg.panel_domain_quota),
                    x,
                )
                for x in remaining
            ]
            scored_remaining.sort(
                key=lambda pair: (
                    -float(pair[0]),
                    -float(pair[1].get("_duration_days", 0.0)),
                    -float(pair[1].get("_volume_total", 0.0)),
                    str(pair[1].get("market_id", "")),
                )
            )
            useful = [row for score, row in scored_remaining if float(score) > 0.0]
            pre_selected = useful if useful else [row for _score, row in scored_remaining]
            batch = pre_selected[: max(1, int(cfg.materialize_batch_size))]
            for row in batch:
                if len(markets) >= cfg.max_pool_size:
                    break
                mid = row["market_id"]
                if mid in tried_history_ids or mid in kept_market_ids:
                    continue
                tried_history_ids.add(mid)
                yid = row.get("_yes_token_id")
                resolve_dt = parse_dt(row.get("resolve_time")) or utc_now()
                p_yes_final = float(row.get("_p_yes_final", 0.5))

                history: List[Dict[str, Any]] = []
                if yid:
                    h, herr = client.get_price_history(str(yid))
                    if herr:
                        reject_ctr[herr] += 1
                        continue
                    history = h or []

                series = build_daily_series(history, resolve_dt=resolve_dt, p_yes_final=p_yes_final)
                series = compress_series(series, cfg.max_series_points)
                if cfg.min_duration_days > 0 and series_span_days(series) < float(cfg.min_duration_days):
                    reject_ctr["short_series_coverage"] += 1
                    continue

                row["community_time_series"] = series
                row["sampled"] = build_sampled_points(series, resolve_dt, cfg.sample_offsets_days)
                sampled = row.get("sampled", {})
                if not isinstance(sampled, dict) or not sampled:
                    reject_ctr["missing_sampled_points"] += 1
                    continue

                # Balance gate during pool construction:
                # prefer rows that cover currently most under-covered horizons.
                need_hard = False
                ratio_map: Dict[str, float] = {}
                for d in cfg.sample_offsets_days:
                    hk = f"{d}d"
                    tgt = max(1, int(cfg.panel_min_counts.get(d, 0)))
                    cur = int(live_horizon_counts.get(hk, 0))
                    if cur < tgt:
                        need_hard = True
                    ratio_map[hk] = float(cur) / float(tgt)
                gate_min_pool = max(20, int(cfg.min_final_count * 0.08))
                if need_hard and ratio_map and len(markets) >= gate_min_pool:
                    min_ratio = min(ratio_map.values())
                    scarce_horizons = {k for k, v in ratio_map.items() if v <= (min_ratio + 0.05)}
                    row_horizons = {k for k, obj in sampled.items() if isinstance(obj, dict)}
                    if row_horizons.isdisjoint(scarce_horizons):
                        reject_ctr["horizon_balance_skip"] += 1
                        continue

                row.pop("_duration_days", None)
                row.pop("_volume_total", None)
                row.pop("_yes_token_id", None)
                row.pop("_p_yes_final", None)
                markets.append(row)
                kept_market_ids.add(mid)
                for d in cfg.sample_offsets_days:
                    hk = f"{d}d"
                    obj = sampled.get(hk) if isinstance(sampled, dict) else None
                    if isinstance(obj, dict):
                        live_horizon_counts[hk] += 1
                if progress is not None:
                    progress.update(1)
                    progress.set_postfix_str(
                        f"offset={offset}, scanned={len(scanned_ids)}, candidates={len(candidates)}, pool={len(markets)}, empty_streak={empty_page_streak}, recent_days={current_recent_resolve_days}"
                    )

        if len(candidates) == before_candidates and len(markets) == before_markets:
            empty_page_streak += 1
        else:
            empty_page_streak = 0

        panels_now = build_panels(
            markets,
            cfg.sample_offsets_days,
            cfg.panel_domain_quota,
            cfg.min_per_domain,
            cfg.max_domain_ratio,
        )
        goals_met_now, _ = panel_goal_status(
            panels_now,
            cfg.sample_offsets_days,
            cfg.panel_min_counts,
            cfg.min_active_domains,
            cfg.min_per_domain,
            cfg.max_domain_ratio,
        )
        final_count_now = final_count_from_panels(panels_now)
        if goals_met_now and final_count_now >= cfg.min_final_count:
            stop_reason = "coverage_and_min_final_reached"
            break
        offset += cfg.page_limit

    if progress is not None:
        progress.close()
    if scan_progress is not None:
        scan_progress.close()
    markets = sorted(markets, key=lambda x: str(x.get("market_id", "")))
    final_markets, selection_report = global_select_markets(markets, cfg)

    # Keep panel stats as diagnostics (not primary selection mechanism anymore).
    panels = build_panels(
        final_markets,
        cfg.sample_offsets_days,
        cfg.panel_domain_quota,
        cfg.min_per_domain,
        cfg.max_domain_ratio,
    )
    coverage_goals_met, coverage_goal_status = panel_goal_status(
        panels,
        cfg.sample_offsets_days,
        cfg.panel_min_counts,
        cfg.min_active_domains,
        cfg.min_per_domain,
        cfg.max_domain_ratio,
    )
    coverage_and_count_goals_met = bool(coverage_goals_met and (len(final_markets) >= cfg.min_final_count))

    payload = {
        "metadata": {
            "generated_at": to_iso(utc_now()),
            "count": len(final_markets),
            "sample_offsets_days": cfg.sample_offsets_days,
            "stop_reason": stop_reason,
            "last_offset": offset,
            "empty_page_streak": empty_page_streak,
            "max_runtime_sec": cfg.max_runtime_sec,
            "runtime_soft_cap_sec": runtime_soft_cap_sec,
            "runtime_hard_cap_sec": runtime_hard_cap_sec,
            "runtime_extended": runtime_extended,
            "max_empty_pages": cfg.max_empty_pages,
            "panel_domain_quota": cfg.panel_domain_quota,
            "pool_count": len(markets),
            "max_pool_size": cfg.max_pool_size,
            "min_final_count": cfg.min_final_count,
            "recent_resolve_days_initial": cfg.recent_resolve_days,
            "recent_resolve_days_effective": current_recent_resolve_days,
            "max_recent_resolve_days": cfg.max_recent_resolve_days,
            "expand_resolve_step_days": cfg.expand_resolve_step_days,
            "auto_expand_resolve_window": cfg.auto_expand_resolve_window,
            "recent_window_expansions": recent_window_expansions,
            "panel_min_counts": {str(k): int(v) for k, v in sorted(cfg.panel_min_counts.items())},
            "min_active_domains": cfg.min_active_domains,
            "min_per_domain": cfg.min_per_domain,
            "max_domain_ratio": cfg.max_domain_ratio,
            "coverage_goals_met": coverage_goals_met,
            "coverage_and_count_goals_met": coverage_and_count_goals_met,
            "coverage_goal_status": coverage_goal_status,
            "selection_report": selection_report,
            "strict_mode": cfg.strict_mode,
        },
        "markets": final_markets,
        "panels": panels,
    }
    return payload, reject_ctr


def parse_args(argv: Optional[Sequence[str]] = None) -> Config:
    p = argparse.ArgumentParser(
        description="Build resolved binary Polymarket question set (adaptive balanced selection)"
    )
    p.add_argument(
        "--recent-resolve-days",
        type=int,
        default=DEFAULT_RECENT_RESOLVE_DAYS,
        help="Manual resolve-time window in days (no auto expansion).",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="src/questions",
        help="Output directory for q_*.json and diagnostics.",
    )
    p.add_argument(
        "--strict",
        dest="strict_mode",
        action="store_true",
        help="Fail if hard coverage constraints are not met.",
    )
    p.add_argument(
        "--best-effort",
        dest="strict_mode",
        action="store_false",
        help="Allow output with gap report when constraints are not fully met.",
    )
    p.set_defaults(strict_mode=DEFAULT_STRICT_MODE)

    # Hidden engineering overrides (kept for internal debugging).
    p.add_argument("--page-limit", type=int, default=DEFAULT_PAGE_LIMIT, help=argparse.SUPPRESS)
    p.add_argument("--min-volume-total", type=float, default=DEFAULT_MIN_VOLUME_TOTAL, help=argparse.SUPPRESS)
    p.add_argument("--validate-n", type=int, default=DEFAULT_VALIDATE_N, help=argparse.SUPPRESS)
    p.add_argument("--sample-offsets", type=str, default=",".join(str(x) for x in DEFAULT_SAMPLE_OFFSETS_DAYS), help=argparse.SUPPRESS)
    p.add_argument("--min-duration-days", type=int, default=DEFAULT_MIN_DURATION_DAYS, help=argparse.SUPPRESS)
    p.add_argument("--max-recent-resolve-days", type=int, default=DEFAULT_MAX_RECENT_RESOLVE_DAYS, help=argparse.SUPPRESS)
    p.add_argument("--expand-resolve-step-days", type=int, default=DEFAULT_EXPAND_RESOLVE_STEP_DAYS, help=argparse.SUPPRESS)
    p.add_argument("--max-series-points", type=int, default=DEFAULT_MAX_SERIES_POINTS, help=argparse.SUPPRESS)
    p.add_argument("--max-runtime-sec", type=int, default=DEFAULT_MAX_RUNTIME_SEC, help=argparse.SUPPRESS)
    p.add_argument("--max-empty-pages", type=int, default=DEFAULT_MAX_EMPTY_PAGES, help=argparse.SUPPRESS)
    p.add_argument("--panel-domain-quota", type=int, default=DEFAULT_PANEL_DOMAIN_QUOTA, help=argparse.SUPPRESS)
    p.add_argument("--panel-min-counts", type=str, default=DEFAULT_PANEL_MIN_COUNTS, help=argparse.SUPPRESS)
    p.add_argument("--min-active-domains", type=int, default=DEFAULT_MIN_ACTIVE_DOMAINS, help=argparse.SUPPRESS)
    p.add_argument("--min-per-domain", type=int, default=DEFAULT_MIN_PER_DOMAIN, help=argparse.SUPPRESS)
    p.add_argument("--max-domain-ratio", type=float, default=DEFAULT_MAX_DOMAIN_RATIO, help=argparse.SUPPRESS)
    p.add_argument("--max-pool-size", type=int, default=DEFAULT_MAX_POOL_SIZE, help=argparse.SUPPRESS)
    p.add_argument("--min-final-count", type=int, default=DEFAULT_MIN_FINAL_COUNT, help=argparse.SUPPRESS)
    p.add_argument("--max-final-count", type=int, default=DEFAULT_MAX_FINAL_COUNT, help=argparse.SUPPRESS)
    p.add_argument("--target-pool-ratio", type=float, default=DEFAULT_TARGET_POOL_RATIO, help=argparse.SUPPRESS)
    p.add_argument("--selection-stability-patience", type=int, default=DEFAULT_SELECTION_STABILITY_PATIENCE, help=argparse.SUPPRESS)
    p.add_argument("--domain-target-slots", type=int, default=DEFAULT_DOMAIN_TARGET_SLOTS, help=argparse.SUPPRESS)
    p.add_argument("--diversity-penalty", type=float, default=DEFAULT_DIVERSITY_PENALTY, help=argparse.SUPPRESS)
    p.add_argument("--materialize-batch-size", type=int, default=DEFAULT_MATERIALIZE_BATCH_SIZE, help=argparse.SUPPRESS)

    args = p.parse_args(argv)

    offsets: List[int] = []
    for x in str(args.sample_offsets).split(","):
        x = x.strip()
        if not x:
            continue
        offsets.append(int(x))
    if not offsets:
        offsets = list(DEFAULT_SAMPLE_OFFSETS_DAYS)

    panel_min_counts: Dict[int, int] = {}
    for part in str(args.panel_min_counts).split(","):
        part = part.strip()
        if not part or ":" not in part:
            continue
        left, right = part.split(":", 1)
        left = left.strip().lower().removesuffix("d")
        right = right.strip()
        if not left.isdigit() or not right.isdigit():
            continue
        panel_min_counts[int(left)] = max(0, int(right))
    if not panel_min_counts:
        panel_min_counts = {30: 80, 7: 150, 1: 150}
    for d in offsets:
        panel_min_counts.setdefault(int(d), 0)

    min_final_count = max(1, int(args.min_final_count))

    return Config(
        page_limit=args.page_limit,
        min_volume_total=args.min_volume_total,
        validate_n=args.validate_n,
        sample_offsets_days=sorted(set(offsets), reverse=True),
        min_duration_days=args.min_duration_days,
        recent_resolve_days=max(1, int(args.recent_resolve_days)),
        max_recent_resolve_days=max(
            max(1, int(args.recent_resolve_days)),
            int(args.max_recent_resolve_days),
        ),
        expand_resolve_step_days=max(1, int(args.expand_resolve_step_days)),
        auto_expand_resolve_window=False,
        max_series_points=max(31, int(args.max_series_points)),
        max_runtime_sec=max(60, int(args.max_runtime_sec)),
        max_empty_pages=max(1, int(args.max_empty_pages)),
        panel_domain_quota=max(1, int(args.panel_domain_quota)),
        panel_min_counts=dict(sorted(panel_min_counts.items())),
        min_active_domains=max(1, int(args.min_active_domains)),
        min_per_domain=max(1, int(args.min_per_domain)),
        max_domain_ratio=max(1.0, float(args.max_domain_ratio)),
        max_pool_size=max(100, int(args.max_pool_size)),
        min_final_count=min_final_count,
        max_final_count=max(min_final_count, int(args.max_final_count)),
        target_pool_ratio=max(0.05, min(0.95, float(args.target_pool_ratio))),
        selection_stability_patience=max(1, int(args.selection_stability_patience)),
        domain_target_slots=max(1, int(args.domain_target_slots)),
        diversity_penalty=max(0.0, float(args.diversity_penalty)),
        strict_mode=bool(args.strict_mode),
        materialize_batch_size=max(1, int(args.materialize_batch_size)),
        output_dir=Path(args.output_dir),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    random.seed(RANDOM_SEED)
    cfg = parse_args(argv)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    data, reject_ctr = build_dataset(cfg)
    markets = data.get("markets", [])

    validate_schema(markets, cfg.sample_offsets_days, cfg.validate_n)

    ts = utc_now().strftime("%Y%m%dT%H%M%SZ")
    out_path = cfg.output_dir / f"q_{ts}.json"
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    viz_paths = make_visualizations(markets, cfg.output_dir, ts)

    domain_counts = Counter(m.get("domain", "other") for m in markets)
    difficult_counts = Counter()
    for m in markets:
        for obj in (m.get("sampled") or {}).values():
            if isinstance(obj, dict):
                difficult_counts[str(obj.get("difficult", "unknown"))] += 1
    horizon_stats = horizon_validity_summary(markets, cfg.sample_offsets_days)
    metadata = data.get("metadata", {}) if isinstance(data, dict) else {}
    selection_report = metadata.get("selection_report", {}) if isinstance(metadata, dict) else {}
    panel_stats = {}
    for k, panel in (data.get("panels") or {}).items():
        if not isinstance(panel, dict):
            continue
        panel_stats[k] = {
            "count": int(panel.get("count", 0)),
            "domain_counts": panel.get("domain_counts", {}),
        }

    print("=== Dataset Summary ===")
    print("count=", metadata.get("count", 0))
    print("stop_reason=", metadata.get("stop_reason"))
    print("min_final_count=", metadata.get("min_final_count"))
    print("domain_counts=", dict(sorted(domain_counts.items())))
    print("difficult_counts=", dict(sorted(difficult_counts.items())))
    print("horizon_validity=", horizon_stats)
    print("panel_stats=", panel_stats)
    print("coverage_goals_met=", metadata.get("coverage_goals_met"))
    print("coverage_and_count_goals_met=", metadata.get("coverage_and_count_goals_met"))
    print("coverage_goal_status=", metadata.get("coverage_goal_status"))
    print(
        "selection_report_summary=",
        {
            "selected_count": selection_report.get("selected_count"),
            "target_count": selection_report.get("target_count"),
            "pool_count": selection_report.get("pool_count"),
            "pool_horizon_counts": selection_report.get("pool_horizon_counts"),
            "hard_constraints_met": selection_report.get("hard_constraints_met"),
            "impossible_hard_constraints": selection_report.get("impossible_hard_constraints"),
            "global_domain_targets": selection_report.get("global_domain_targets"),
            "global_domain_counts": selection_report.get("global_domain_counts"),
            "global_domain_gaps": selection_report.get("global_domain_gaps"),
            "impossible_domain_constraints": selection_report.get("impossible_domain_constraints"),
            "effective_policy": selection_report.get("effective_policy"),
            "stopped_by_stability": selection_report.get("stopped_by_stability"),
        },
    )
    print("top_reject_reasons=", reject_ctr.most_common(10))
    print("output_main=", out_path)
    if viz_paths:
        print("visualizations=", [str(p) for p in viz_paths])
    else:
        print("visualizations= []")

    hard_ok = bool(selection_report.get("hard_constraints_met", False))
    count_ok = int(metadata.get("count", 0)) >= int(metadata.get("min_final_count", cfg.min_final_count))
    if cfg.strict_mode and (not hard_ok or not count_ok):
        print("strict_mode_failure=constraints_not_met")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
