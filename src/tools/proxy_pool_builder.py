"""Build a high-quality proxy pool from raw candidates.

Usage example:
    PYTHONPATH=src python src/tools/proxy_pool_builder.py \
      --input /path/to/freeproxy.json \
      --output config/proxy_pool/high_quality_proxies.json \
      --urls "https://httpbin.org/ip,https://www.reuters.com/" \
      --concurrency 40
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote

import httpx


def _expand_raw_proxy_items(raw: Any) -> list[Any]:
    if isinstance(raw, list):
        return list(raw)
    if not isinstance(raw, dict):
        return []
    if isinstance(raw.get("proxies"), list):
        return list(raw.get("proxies") or [])

    items: list[Any] = []
    for value in raw.values():
        if isinstance(value, list):
            items.extend(value)
        elif isinstance(value, dict):
            items.append(value)
    return items


def _extract_proxy_spec(item: Any) -> Optional[str]:
    if isinstance(item, str):
        spec = item.strip()
        return spec or None
    if not isinstance(item, dict):
        return None

    direct = str(item.get("proxy") or item.get("proxy_spec") or item.get("spec") or item.get("proxy_url") or "").strip()
    if direct:
        return direct

    host = str(item.get("ip") or item.get("host") or item.get("proxy_ip") or "").strip()
    port = str(item.get("port") or item.get("proxy_port") or "").strip()
    if not host or not port:
        return None

    username = str(item.get("user") or item.get("username") or "").strip()
    password = str(item.get("pass") or item.get("password") or "").strip()
    if username and password:
        return f"{host}:{port}:{username}:{password}"
    return f"{host}:{port}"


def _proxy_spec_to_httpx_url(proxy_spec: str, *, scheme: str = "http") -> Optional[str]:
    raw = str(proxy_spec or "").strip()
    if not raw:
        return None
    if "://" in raw:
        return raw

    parts = raw.split(":")
    if len(parts) < 2:
        return None
    host = parts[0].strip()
    port = parts[1].strip()
    if not host or not port:
        return None

    auth = ""
    if len(parts) >= 4:
        username = quote(parts[2], safe="")
        password = quote(":".join(parts[3:]), safe="")
        auth = f"{username}:{password}@"

    return f"{scheme}://{auth}{host}:{port}"


@dataclass
class ProxyProbeResult:
    proxy: str
    attempts: int
    successes: int
    success_rate: float
    mean_latency_sec: Optional[float]
    quality_score: float
    error_top: str


def _quality_score(*, success_rate: float, mean_latency_sec: Optional[float], max_latency_sec: float) -> float:
    if mean_latency_sec is None:
        speed_score = 0.0
    else:
        speed_score = max(0.0, min(1.0, 1.0 - (mean_latency_sec / max(0.01, max_latency_sec))))
    # Success dominates; speed is secondary.
    return round((success_rate * 0.75) + (speed_score * 0.25), 4)


async def _probe_single_proxy(
    proxy_spec: str,
    *,
    test_urls: list[str],
    attempts_per_url: int,
    connect_timeout: float,
    read_timeout: float,
    max_latency_sec: float,
) -> ProxyProbeResult:
    proxy_url = _proxy_spec_to_httpx_url(proxy_spec)
    attempts = 0
    successes = 0
    latencies: list[float] = []
    errors: Counter[str] = Counter()

    if not proxy_url:
        return ProxyProbeResult(
            proxy=proxy_spec,
            attempts=0,
            successes=0,
            success_rate=0.0,
            mean_latency_sec=None,
            quality_score=0.0,
            error_top="invalid_proxy_spec",
        )

    timeout = httpx.Timeout(read_timeout, connect=connect_timeout)
    headers = {"User-Agent": "future-forecasting-proxy-prober/1.0"}

    try:
        async with httpx.AsyncClient(
            proxy=proxy_url,
            timeout=timeout,
            follow_redirects=True,
            trust_env=False,
            headers=headers,
        ) as client:
            for _ in range(max(1, attempts_per_url)):
                for url in test_urls:
                    attempts += 1
                    start = time.perf_counter()
                    try:
                        response = await client.get(url)
                    except Exception as exc:
                        errors[type(exc).__name__] += 1
                        continue
                    cost = time.perf_counter() - start
                    if 200 <= int(response.status_code) < 400:
                        successes += 1
                        latencies.append(cost)
                    else:
                        errors[f"http_{int(response.status_code)}"] += 1
    except Exception as exc:
        errors[type(exc).__name__] += 1

    success_rate = (successes / attempts) if attempts > 0 else 0.0
    mean_latency_sec = (sum(latencies) / len(latencies)) if latencies else None
    score = _quality_score(
        success_rate=success_rate,
        mean_latency_sec=mean_latency_sec,
        max_latency_sec=max_latency_sec,
    )
    error_top = errors.most_common(1)[0][0] if errors else ""

    return ProxyProbeResult(
        proxy=proxy_spec,
        attempts=attempts,
        successes=successes,
        success_rate=round(success_rate, 4),
        mean_latency_sec=round(mean_latency_sec, 4) if mean_latency_sec is not None else None,
        quality_score=score,
        error_top=error_top,
    )


async def _probe_all_proxies(
    proxy_specs: list[str],
    *,
    test_urls: list[str],
    attempts_per_url: int,
    connect_timeout: float,
    read_timeout: float,
    max_latency_sec: float,
    concurrency: int,
) -> list[ProxyProbeResult]:
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def _wrapped(spec: str) -> ProxyProbeResult:
        async with semaphore:
            return await _probe_single_proxy(
                spec,
                test_urls=test_urls,
                attempts_per_url=attempts_per_url,
                connect_timeout=connect_timeout,
                read_timeout=read_timeout,
                max_latency_sec=max_latency_sec,
            )

    tasks = [asyncio.create_task(_wrapped(spec)) for spec in proxy_specs]
    results: list[ProxyProbeResult] = []
    done = 0
    for task in asyncio.as_completed(tasks):
        result = await task
        results.append(result)
        done += 1
        if done % 20 == 0 or done == len(tasks):
            print(f"[probe] finished {done}/{len(tasks)}")
    return results


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build high-quality proxy pool JSON")
    p.add_argument("--input", required=True, help="Raw proxy JSON path")
    p.add_argument("--output", required=True, help="Output high-quality proxy JSON path")
    p.add_argument(
        "--urls",
        default="https://httpbin.org/ip",
        help="Comma-separated probe URLs, e.g. https://httpbin.org/ip,https://www.reuters.com/",
    )
    p.add_argument("--attempts-per-url", type=int, default=1, help="Probe attempts per URL for each proxy")
    p.add_argument("--concurrency", type=int, default=40, help="Concurrent proxy probes")
    p.add_argument("--connect-timeout", type=float, default=4.0, help="Connect timeout seconds")
    p.add_argument("--read-timeout", type=float, default=8.0, help="Read timeout seconds")
    p.add_argument("--min-success-rate", type=float, default=0.70, help="Keep proxies with success_rate >= value")
    p.add_argument(
        "--max-mean-latency-sec",
        type=float,
        default=4.0,
        help="Keep proxies with mean latency <= value",
    )
    p.add_argument("--top-k", type=int, default=300, help="Keep top-K by quality score after filtering")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    urls = [x.strip() for x in str(args.urls).split(",") if x.strip()]
    if not input_path.exists():
        raise FileNotFoundError(f"input file not found: {input_path}")
    if not urls:
        raise ValueError("at least one probe URL is required")

    raw = json.loads(input_path.read_text(encoding="utf-8"))
    proxy_specs = []
    for item in _expand_raw_proxy_items(raw):
        spec = _extract_proxy_spec(item)
        if spec:
            proxy_specs.append(spec)
    proxy_specs = list(dict.fromkeys(proxy_specs))
    if not proxy_specs:
        raise RuntimeError("no proxy candidates loaded from input")

    print(f"[load] candidates={len(proxy_specs)} urls={len(urls)}")
    results = asyncio.run(
        _probe_all_proxies(
            proxy_specs,
            test_urls=urls,
            attempts_per_url=max(1, int(args.attempts_per_url)),
            connect_timeout=float(args.connect_timeout),
            read_timeout=float(args.read_timeout),
            max_latency_sec=float(args.max_mean_latency_sec),
            concurrency=max(1, int(args.concurrency)),
        )
    )

    kept = [
        item for item in results
        if item.success_rate >= float(args.min_success_rate)
        and item.mean_latency_sec is not None
        and item.mean_latency_sec <= float(args.max_mean_latency_sec)
    ]
    kept.sort(key=lambda x: (x.quality_score, x.success_rate, -(x.mean_latency_sec or 9999.0)), reverse=True)
    top_k = max(1, int(args.top_k))
    kept = kept[:top_k]

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_candidates": len(proxy_specs),
        "tested_candidates": len(results),
        "kept_candidates": len(kept),
        "test_urls": urls,
        "min_success_rate": float(args.min_success_rate),
        "max_mean_latency_sec": float(args.max_mean_latency_sec),
        "proxies": [
            {
                "proxy": item.proxy,
                "quality_score": item.quality_score,
                "success_rate": item.success_rate,
                "mean_latency_sec": item.mean_latency_sec,
                "successes": item.successes,
                "attempts": item.attempts,
                "error_top": item.error_top,
            }
            for item in kept
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[done] wrote {len(kept)} high-quality proxies to {output_path} "
        f"(from {len(proxy_specs)} candidates)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
