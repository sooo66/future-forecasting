#!/usr/bin/env python3
import json
from pathlib import Path
from collections import defaultdict

INPUT_PATH = Path(
    "/Users/ymx66/Workspace/future-forecasting/data/url_pool/url_pool.jsonl"
)
OUTPUT_PATH = INPUT_PATH.with_name("url_pool_filter.jsonl")

# 需要过滤掉的站点
EXCLUDED_DOMAINS = {
    "reuters.com",
    "bloomberg.com",
    "wsj.com",
    "nytimes.com",
    "forbes.com",
    "theepochtimes.com",
}

MAX_URLS_PER_DOMAIN = 5


def _normalize_domain(domain: str) -> str:
    """简单标准化域名，去掉前缀 www. 并转小写。"""
    domain = domain.lower()
    if domain.startswith("www."):
        domain = domain[4:]
    return domain


def main() -> None:
    # 记录每个站点已经写出的 URL 数量（按标准化后的域名统计）
    counts_by_domain = defaultdict(int)

    total_input = 0
    total_written = 0

    with INPUT_PATH.open("r", encoding="utf-8") as fin, OUTPUT_PATH.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # 跳过坏行
                continue

            total_input += 1

            url = obj.get("url")
            domain = obj.get("domain")
            if not url or not domain:
                continue

            normalized_domain = _normalize_domain(domain)

            # 过滤掉指定站点
            if normalized_domain in EXCLUDED_DOMAINS:
                continue

            # 每个站点最多保留 5 条
            if counts_by_domain[normalized_domain] >= MAX_URLS_PER_DOMAIN:
                continue

            counts_by_domain[normalized_domain] += 1

            # 直接写原始对象，保持其他 key 不变
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            total_written += 1

    print(f"Total input URLs: {total_input}")
    print(f"Domains kept: {len(counts_by_domain)}")
    print(f"Total URLs written (<= {MAX_URLS_PER_DOMAIN} per site): {total_written}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()


