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
import math
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # visualization is best-effort
    plt = None


GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"

# ---- defaults ----
DEFAULT_PAGE_LIMIT = 200
DEFAULT_MAX_MARKETS = 1800
DEFAULT_MIX_MAX_OFFSET = 12000
DEFAULT_MIN_LIQUIDITY = 0.0
DEFAULT_MIN_VOLUME_TOTAL = 100.0
DEFAULT_MIN_VOLUME_24H = 0.0
DEFAULT_MIN_CRITERIA_CHARS = 40
DEFAULT_SUM_TOLERANCE = 0.08
DEFAULT_SAMPLE_SIZE = 300
DEFAULT_VALIDATE_N = 20
DEFAULT_SAMPLE_OFFSETS_DAYS = [30, 7, 1]
DEFAULT_MIN_DURATION_DAYS = 30
DEFAULT_MAX_SERIES_POINTS = 180

REQUEST_TIMEOUT = 25
MAX_RETRIES = 4
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
    max_markets: int
    page_limit: int
    mix_max_offset: int
    min_liquidity: float
    min_volume_total: float
    min_volume_24h: float
    min_criteria_chars: int
    sum_tolerance: float
    sample_size: int
    validate_n: int
    sample_offsets_days: List[int]
    min_duration_days: int
    max_series_points: int
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


def interleaved_offsets(max_offset: int, page_limit: int) -> List[int]:
    pages = list(range(0, max_offset + 1, page_limit))
    out: List[int] = []
    i, j = 0, len(pages) - 1
    while i <= j:
        out.append(pages[i])
        i += 1
        if i <= j:
            out.append(pages[j])
            j -= 1
    return out


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

    def get_price_history(self, yes_token_id: str) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
        params = {
            "market": yes_token_id,
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


def is_yes_no(outcomes: List[str]) -> bool:
    return len(outcomes) == 2 and {x.lower() for x in outcomes} == {"yes", "no"}


def resolved_status(m: Dict[str, Any], prices: List[float]) -> bool:
    if str(m.get("umaResolutionStatus") or "").strip().lower() == "resolved":
        return True
    if not bool(m.get("closed")) or len(prices) != 2:
        return False
    return max(prices) >= 0.95 and min(prices) <= 0.05


def valid_probability_pair(prices: List[float], tol: float) -> bool:
    if len(prices) != 2:
        return False
    if any(p < -1e-9 or p > 1 + 1e-9 for p in prices):
        return False
    return abs((prices[0] + prices[1]) - 1.0) <= tol


def extract_answer_yes(outcomes: List[str], prices: List[float]) -> Optional[int]:
    if len(outcomes) != 2 or len(prices) != 2:
        return None
    try:
        yes_idx = [x.lower() for x in outcomes].index("yes")
    except ValueError:
        return None
    return 1 if prices[yes_idx] >= 0.5 else 0


def yes_token_id(m: Dict[str, Any], outcomes: List[str]) -> Optional[str]:
    ids = maybe_json_list(m.get("clobTokenIds"))
    if len(ids) < 2:
        return None
    try:
        yes_idx = [x.lower() for x in outcomes].index("yes")
    except ValueError:
        return None
    if yes_idx >= len(ids):
        return None
    return str(ids[yes_idx])


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
    return desc


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
        return "crypto"
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

    if max_points < 2:
        return [series[0], series[-1]]

    step = (len(series) - 1) / float(max_points - 1)
    idxs = sorted({int(round(i * step)) for i in range(max_points)})
    out = [series[i] for i in idxs]
    if out[0]["t"] != series[0]["t"]:
        out[0] = series[0]
    if out[-1]["t"] != series[-1]["t"]:
        out[-1] = series[-1]
    return out


def series_span_days(series: List[Dict[str, Any]]) -> float:
    if len(series) < 2:
        return 0.0
    t0 = parse_dt(series[0].get("t"))
    t1 = parse_dt(series[-1].get("t"))
    if not t0 or not t1:
        return 0.0
    return max((t1 - t0).total_seconds() / 86400.0, 0.0)


def nearest_prob_at_or_before(series: List[Dict[str, Any]], target: datetime) -> Optional[float]:
    target_iso = to_iso(target)
    before = [x for x in series if x.get("t") and x["t"] <= target_iso]
    if before:
        return to_float(before[-1].get("p_yes"))
    after = [x for x in series if x.get("t") and x["t"] > target_iso]
    if after:
        return to_float(after[0].get("p_yes"))
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
    vol_norm = min(max(vol / 0.08, 0.0), 1.0)
    score = 0.75 * uncertainty + 0.25 * vol_norm
    if score >= 0.66:
        return "hard"
    if score >= 0.38:
        return "medium"
    return "easy"


def build_sampled_points(series: List[Dict[str, Any]], resolve_dt: datetime, offsets: List[int]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for d in offsets:
        key = f"{d}d"
        target = resolve_dt - timedelta(days=d)
        p = nearest_prob_at_or_before(series, target)
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
        if not isinstance(s, dict):
            errors += 1
            continue
        for d in sample_offsets_days:
            k = f"{d}d"
            if k not in s:
                errors += 1
                continue
            obj = s[k]
            if not isinstance(obj, dict):
                errors += 1
                continue
            if "p_yes" not in obj or "difficult" not in obj or "t" not in obj:
                errors += 1

    print(f"schema_check_samples={n} errors={errors}")


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

    return paths


def build_dataset(cfg: Config) -> Tuple[Dict[str, Any], Counter]:
    client = PolymarketClient()
    reject_ctr: Counter = Counter()

    candidates: List[Dict[str, Any]] = []
    seen_ids = set()
    scanned = 0

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

    offsets = interleaved_offsets(cfg.mix_max_offset, cfg.page_limit)

    for offset in offsets:
        if scanned >= cfg.max_markets:
            break

        for _src_domain, src_tag_id in domain_sources:
            if scanned >= cfg.max_markets:
                break

            page, err = client.list_markets_page(limit=cfg.page_limit, offset=offset, tag_id=(src_tag_id or None))
            if err:
                reject_ctr[err] += 1
                continue
            if not page:
                continue

            for m in page:
                if scanned >= cfg.max_markets:
                    break
                mid = market_id(m)
                if not mid or mid in seen_ids:
                    continue
                seen_ids.add(mid)
                scanned += 1

                outcomes, prices = parse_outcomes(m)
                if not resolved_status(m, prices):
                    reject_ctr["unresolved"] += 1
                    continue
                if not is_yes_no(outcomes):
                    reject_ctr["invalid_market_data"] += 1
                    continue

                open_dt, close_dt, resolve_dt = parse_times(m)
                if open_dt is None or close_dt is None or resolve_dt is None:
                    reject_ctr["missing_resolution"] += 1
                    continue

                if duration_days(open_dt, close_dt) <= cfg.min_duration_days:
                    reject_ctr["short_duration"] += 1
                    continue

                crit = resolution_criteria(m)
                if len(crit.strip()) < cfg.min_criteria_chars:
                    reject_ctr["missing_criteria"] += 1
                    continue

                liquidity = to_float(m.get("liquidityNum"))
                if liquidity is None:
                    liquidity = to_float(m.get("liquidity"))
                volume_total = to_float(m.get("volumeNum"))
                if volume_total is None:
                    volume_total = to_float(m.get("volume"))
                volume_24h = to_float(m.get("volume24hr"))

                if (liquidity or 0.0) < cfg.min_liquidity:
                    reject_ctr["low_liquidity"] += 1
                    continue
                if (volume_total or 0.0) < cfg.min_volume_total or (volume_24h or 0.0) < cfg.min_volume_24h:
                    reject_ctr["low_volume"] += 1
                    continue

                if not valid_probability_pair(prices, cfg.sum_tolerance):
                    reject_ctr["invalid_market_data"] += 1
                    continue

                ans = extract_answer_yes(outcomes, prices)
                if ans is None:
                    reject_ctr["missing_resolution"] += 1
                    continue

                try:
                    yes_idx = [x.lower() for x in outcomes].index("yes")
                except ValueError:
                    reject_ctr["missing_resolution"] += 1
                    continue

                p_yes_final = float(prices[yes_idx])
                yid = yes_token_id(m, outcomes)
                domain = infer_domain(m)

                candidates.append(
                    {
                        "market_id": mid,
                        "question": str(m.get("question") or "").strip(),
                        "domain": domain,
                        "description": str(m.get("description") or "").strip() or None,
                        "resolution_criteria": crit,
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

    selected = balanced_sample_by_domain(candidates, cfg.sample_size)

    markets: List[Dict[str, Any]] = []
    for row in selected:
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
        if series_span_days(series) <= cfg.min_duration_days:
            reject_ctr["short_series_coverage"] += 1
            continue

        row["community_time_series"] = series
        row["sampled"] = build_sampled_points(series, resolve_dt, cfg.sample_offsets_days)

        row.pop("_duration_days", None)
        row.pop("_volume_total", None)
        row.pop("_yes_token_id", None)
        row.pop("_p_yes_final", None)
        markets.append(row)

    payload = {
        "metadata": {
            "generated_at": to_iso(utc_now()),
            "count": len(markets),
            "sample_offsets_days": cfg.sample_offsets_days,
        },
        "markets": markets,
    }
    return payload, reject_ctr


def parse_args(argv: Optional[Sequence[str]] = None) -> Config:
    p = argparse.ArgumentParser(description="Build resolved binary Polymarket dataset")
    p.add_argument("--max-markets", type=int, default=DEFAULT_MAX_MARKETS)
    p.add_argument("--page-limit", type=int, default=DEFAULT_PAGE_LIMIT)
    p.add_argument("--mix-max-offset", type=int, default=DEFAULT_MIX_MAX_OFFSET)
    p.add_argument("--min-liquidity", type=float, default=DEFAULT_MIN_LIQUIDITY)
    p.add_argument("--min-volume-total", type=float, default=DEFAULT_MIN_VOLUME_TOTAL)
    p.add_argument("--min-volume-24h", type=float, default=DEFAULT_MIN_VOLUME_24H)
    p.add_argument("--min-criteria-chars", type=int, default=DEFAULT_MIN_CRITERIA_CHARS)
    p.add_argument("--sum-tolerance", type=float, default=DEFAULT_SUM_TOLERANCE)
    p.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    p.add_argument("--validate-n", type=int, default=DEFAULT_VALIDATE_N)
    p.add_argument("--sample-offsets", type=str, default=",".join(str(x) for x in DEFAULT_SAMPLE_OFFSETS_DAYS))
    p.add_argument("--min-duration-days", type=int, default=DEFAULT_MIN_DURATION_DAYS)
    p.add_argument("--max-series-points", type=int, default=DEFAULT_MAX_SERIES_POINTS)
    p.add_argument("--output-dir", type=str, default="src/questions")

    args = p.parse_args(argv)

    offsets: List[int] = []
    for x in str(args.sample_offsets).split(","):
        x = x.strip()
        if not x:
            continue
        offsets.append(int(x))
    if not offsets:
        offsets = list(DEFAULT_SAMPLE_OFFSETS_DAYS)

    return Config(
        max_markets=args.max_markets,
        page_limit=args.page_limit,
        mix_max_offset=args.mix_max_offset,
        min_liquidity=args.min_liquidity,
        min_volume_total=args.min_volume_total,
        min_volume_24h=args.min_volume_24h,
        min_criteria_chars=args.min_criteria_chars,
        sum_tolerance=args.sum_tolerance,
        sample_size=args.sample_size,
        validate_n=args.validate_n,
        sample_offsets_days=sorted(set(offsets), reverse=True),
        min_duration_days=args.min_duration_days,
        max_series_points=max(31, int(args.max_series_points)),
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

    print("=== Dataset Summary ===")
    print("count=", data.get("metadata", {}).get("count", 0))
    print("domain_counts=", dict(sorted(domain_counts.items())))
    print("difficult_counts=", dict(sorted(difficult_counts.items())))
    print("top_reject_reasons=", reject_ctr.most_common(10))
    print("output_main=", out_path)
    if viz_paths:
        print("visualizations=", [str(p) for p in viz_paths])
    else:
        print("visualizations= []")

    return 0


if __name__ == "__main__":
    sys.exit(main())
