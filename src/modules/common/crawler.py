"""简单可用的 Crawl4AI 爬虫实现

- 继承自 ``crawl4ai.hub.BaseCrawler``，对外暴露单一的 ``run`` 方法；
- 依赖 ``Config``，但支持直接传入 dict，方便在测试中构造临时配置；
- 输出与 ``utils.models.Record`` 对齐：每条爬取结果都序列化为一条 JSONL 记录。

当前目标是「能跑且易测」，未特别优化扩展性或高级反爬策略。
"""
from __future__ import annotations

import asyncio
from collections import Counter
import json
import os
import random
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional
from urllib.parse import urlparse

import httpx
from loguru import logger

from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CacheMode,
    CrawlerRunConfig,
    ProxyConfig,
    UndetectedAdapter,
)
from crawl4ai.async_crawler_strategy import AsyncPlaywrightCrawlerStrategy
from crawl4ai.async_dispatcher import MemoryAdaptiveDispatcher
from crawl4ai.async_logger import AsyncLogger
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.proxy_strategy import RoundRobinProxyStrategy

try:  # 型检查 / 运行时都尽量使用真实 BaseCrawler
    from crawl4ai.hub import BaseCrawler
except Exception:  # pragma: no cover - 旧版本 crawl4ai 没有 hub 时兜底
    class BaseCrawler:  # type: ignore[override]
        async def run(self, *args: Any, **kwargs: Any) -> Any:  # noqa: D401
            """Placeholder BaseCrawler when crawl4ai.hub is unavailable."""
            raise RuntimeError("crawl4ai.hub.BaseCrawler is not available")

from modules.common.proxy_pool import ProxyManager
from modules.common.extractor import Extractor
from utils.config import Config
from utils.models import Record

try:
    import trafilatura
except Exception:  # pragma: no cover - optional dependency fallback
    trafilatura = None  # type: ignore[assignment]

try:
    from scrapling import AsyncFetcher
except Exception:  # pragma: no cover - scrapling optional at runtime
    AsyncFetcher = None  # type: ignore[assignment]


HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+\S+")


@dataclass
class _MdBlock:
    heading: Optional[str]
    lines: List[str]
    content_chars: int


class Crawler:
    """基于 Crawl4AI 的新闻爬虫实现。

    用法示例（在代码中/测试中调用）：

    .. code-block:: python

        config_dict = {"general": {"concurrent_crawls": 5}, "paths": {"processed_data_dir": "./data"}}
        crawler = Crawler(config=config_dict, urls=["https://example.com"])
        records = asyncio.run(crawler.run())
    """

    def __init__(
        self,
        config: "Config | dict | str | Path" = "config/settings.toml",
        *,
        source_path: Optional[Path] = None,
        urls: Optional[List[str]] = None,
        output_path: Optional[Path] = None,
        proxy_file: Optional[Path] = None,
        use_proxy: Optional[bool] = None,
    ) -> None:
        # Config 支持 dict，便于测试；字符串/Path 时走原来的 TOML 加载流程
        if isinstance(config, Config):
            self.config = config
        elif isinstance(config, (str, Path, dict)):
            self.config = Config(config)  # type: ignore[arg-type]
        else:
            raise TypeError(f"Unsupported config type: {type(config)!r}")

        self.source_path = source_path
        self.urls = urls
        self.output_path = output_path or self._default_output_path()

        # markdown 原始内容调试文件（与 output_path 同目录，同名加 .raw_md 后缀）
        self._raw_markdown_path = self.output_path.with_suffix(self.output_path.suffix + ".raw_md")

        # 记录 URL -> 元信息（如 published_at），便于透传到 Record
        self._url_meta: dict[str, dict[str, Any]] = {}
        self._last_input_urls: set[str] = set()
        self._last_succeeded_urls: set[str] = set()
        self._last_failed_urls: set[str] = set()
        self._last_failed_reasons: dict[str, str] = {}

        self._enable_scrapling_fallback = bool(self.config.get("crawler.enable_scrapling_fallback", True))
        self._enable_jina_reader_fallback = bool(self.config.get("crawler.enable_jina_reader_fallback", True))
        self._fallback_concurrency = max(
            1,
            int(self.config.get("crawler.fallback_concurrency", self.config.concurrent_crawls)),
        )
        self._fallback_timeout_sec = max(
            5.0,
            float(self.config.get("crawler.fallback_timeout_sec", self.config.browser_timeout)),
        )
        self._jina_reader_prefix = str(self.config.get("crawler.jina_reader_prefix", "https://r.jina.ai/")).strip()
        if not self._jina_reader_prefix:
            self._jina_reader_prefix = "https://r.jina.ai/"
        if not self._jina_reader_prefix.endswith("/"):
            self._jina_reader_prefix += "/"

        raw_primary_engine = str(self.config.get("crawler.primary_engine", "crawl4ai")).strip().lower()
        if raw_primary_engine not in {"crawl4ai", "scrapling", "hybrid"}:
            logger.warning(f"未知 crawler.primary_engine={raw_primary_engine}，回退到 crawl4ai")
            raw_primary_engine = "crawl4ai"
        self._primary_engine = raw_primary_engine
        self._scrapling_primary_concurrency = max(
            1,
            int(self.config.get("crawler.scrapling_primary_concurrency", self.config.concurrent_crawls)),
        )
        self._scrapling_quality_min_chars = max(
            0,
            int(self.config.get("crawler.scrapling_quality_min_chars", 280)),
        )
        self._scrapling_primary_timeout_sec = max(
            3.0,
            float(self.config.get("crawler.scrapling_primary_timeout_sec", 12.0)),
        )
        self._scrapling_primary_retries = max(
            1,
            int(self.config.get("crawler.scrapling_primary_retries", 1)),
        )
        self._scrapling_primary_per_domain_concurrency = max(
            1,
            int(self.config.get("crawler.scrapling_primary_per_domain_concurrency", 2)),
        )
        self._scrapling_primary_delay_min_sec = max(
            0.0,
            float(self.config.get("crawler.scrapling_primary_delay_min_sec", 0.0)),
        )
        self._scrapling_primary_delay_max_sec = max(
            self._scrapling_primary_delay_min_sec,
            float(self.config.get("crawler.scrapling_primary_delay_max_sec", 0.0)),
        )
        self._scrapling_primary_chunk_size = max(
            self._scrapling_primary_concurrency,
            int(
                self.config.get(
                    "crawler.scrapling_primary_chunk_size",
                    max(64, self._scrapling_primary_concurrency * 4),
                )
            ),
        )
        self._scrapling_short_circuit_sample_size = max(
            1,
            int(self.config.get("crawler.scrapling_short_circuit_sample_size", 80)),
        )
        self._scrapling_short_circuit_fail_ratio = min(
            1.0,
            max(0.0, float(self.config.get("crawler.scrapling_short_circuit_fail_ratio", 0.85))),
        )
        self._enable_crawl4ai_rescue_after_scrapling = bool(
            self.config.get("crawler.enable_crawl4ai_rescue_after_scrapling", True)
        )

        if AsyncFetcher is None and (self._primary_engine in {"scrapling", "hybrid"} or self._enable_scrapling_fallback):
            logger.warning(
                "scrapling 未安装或不可用：当前进程将无法使用 scrapling 路径。"
            )
        if AsyncFetcher is None and self._primary_engine in {"scrapling", "hybrid"}:
            logger.warning(
                f"primary_engine={self._primary_engine} 但 scrapling 不可用，自动回退为 crawl4ai"
            )
            self._primary_engine = "crawl4ai"

        self.proxy_manager = ProxyManager(
            proxy_file=proxy_file,
            use_proxy=self.config.use_proxy if use_proxy is None else use_proxy,
        )

        # 从独立配置文件加载域名 -> main content CSS selectors 映射
        # 配置文件路径: config/main_content_selectors.json
        # 对于每个域名，可以同时配置:
        #   - targets:   只抓取这些容器内的正文
        #   - exclude:   显式排除这些容器（如导航栏、推荐区等）
        (
            self._domain_target_selectors,
            self._domain_excluded_selectors,
        ) = self._load_domain_selectors()

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------
    async def run(self) -> List[Record]:  # type: ignore[override]
        """运行爬虫并返回 `Record` 列表。"""
        records: List[Record] = []
        async for record in self.run_stream():
            records.append(record)
        return records

    @property
    def last_succeeded_urls(self) -> set[str]:
        return set(self._last_succeeded_urls)

    @property
    def last_failed_urls(self) -> set[str]:
        return set(self._last_failed_urls)

    @property
    def last_failed_reasons(self) -> dict[str, str]:
        return dict(self._last_failed_reasons)

    async def run_stream(self) -> AsyncIterator[Record]:
        """运行爬虫并流式产出 `Record`。"""
        if self.urls is not None:
            urls = self._normalize_urls_input(self.urls)
        else:
            urls = self._load_urls(self.source_path)
        if not urls:
            self._reset_run_tracking([])
            logger.warning("输入 URL 为空，跳过爬取")
            return

        self._reset_run_tracking(urls)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.proxy_manager.refresh_env()
        logger.debug(
            f"爬虫主引擎: {self._primary_engine} "
            f"(scrapling_qmin={self._scrapling_quality_min_chars}, "
            f"scrapling_timeout={self._scrapling_primary_timeout_sec}s, "
            f"scrapling_retries={self._scrapling_primary_retries}, "
            f"scrapling_domain_q={self._scrapling_primary_per_domain_concurrency}, "
            f"scrapling_delay={self._scrapling_primary_delay_min_sec:.2f}-{self._scrapling_primary_delay_max_sec:.2f}s, "
            f"scrapling_chunk={self._scrapling_primary_chunk_size}, "
            f"scrapling_short_circuit={self._scrapling_short_circuit_sample_size}/"
            f"{self._scrapling_short_circuit_fail_ratio:.2f}, "
            f"crawl4ai_rescue={self._enable_crawl4ai_rescue_after_scrapling})"
        )

        total_records = 0
        if self._primary_engine in {"scrapling", "hybrid"}:
            rescue_urls: List[str] = []
            succeeded_urls: set[str] = set()
            async for record, rescue_url in self._crawl_with_scrapling_primary_stream(urls):
                if record is not None:
                    succeeded_urls.add(record.url)
                    self._mark_success(record.url)
                    total_records += 1
                    yield record
                if rescue_url:
                    rescue_urls.append(rescue_url)

            should_rescue = self._primary_engine == "hybrid" or self._enable_crawl4ai_rescue_after_scrapling
            if rescue_urls and should_rescue:
                logger.debug(f"scrapling 主抓取完成，进入 crawl4ai rescue: {len(rescue_urls)} URLs")
                async for record in self._crawl_with_crawl4ai_stream(rescue_urls, pre_succeeded_urls=succeeded_urls):
                    self._mark_success(record.url)
                    total_records += 1
                    yield record
            elif rescue_urls:
                logger.debug(f"scrapling 主抓取完成，未启用 crawl4ai rescue，丢弃 {len(rescue_urls)} URLs")
        else:
            async for record in self._crawl_with_crawl4ai_stream(urls):
                self._mark_success(record.url)
                total_records += 1
                yield record

        self._last_failed_urls = set(self._last_input_urls) - set(self._last_succeeded_urls)
        self._last_failed_reasons = {
            url: self._last_failed_reasons.get(url, "unknown_failure")
            for url in self._last_failed_urls
        }
        logger.debug(f"爬取完成，总记录数: {total_records}，输出: {self.output_path}")

    def _reset_run_tracking(self, urls: List[str]) -> None:
        self._last_input_urls = {str(url).strip() for url in urls if str(url).strip()}
        self._last_succeeded_urls = set()
        self._last_failed_urls = set()
        self._last_failed_reasons = {}

    def _mark_success(self, url: str) -> None:
        clean = str(url or "").strip()
        if not clean:
            return
        self._last_succeeded_urls.add(clean)
        self._last_failed_reasons.pop(clean, None)

    def _mark_failure(self, url: str, reason: Optional[str]) -> None:
        clean = str(url or "").strip()
        if not clean:
            return
        summary = self._sanitize_failure_reason(reason)
        if summary:
            self._last_failed_reasons[clean] = summary

    @staticmethod
    def _sanitize_failure_reason(reason: Optional[str], *, limit: int = 240) -> str:
        text = re.sub(r"\s+", " ", str(reason or "").strip())
        if not text:
            return ""
        return text[:limit]

    @staticmethod
    def _domain_key_for_url(url: str) -> str:
        try:
            netloc = urlparse(url).netloc or ""
        except Exception:
            netloc = ""
        host = netloc.split("@")[-1].split(":")[0].strip().lower()
        return host or "unknown"

    def _sample_scrapling_primary_delay(self) -> float:
        low = self._scrapling_primary_delay_min_sec
        high = self._scrapling_primary_delay_max_sec
        if high <= 0:
            return 0.0
        if high <= low:
            return low
        return random.uniform(low, high)

    async def _crawl_with_scrapling_primary_stream(
        self,
        urls: List[str],
    ) -> AsyncIterator[tuple[Optional[Record], Optional[str]]]:
        semaphore = asyncio.Semaphore(self._scrapling_primary_concurrency)
        domain_semaphores: dict[str, asyncio.Semaphore] = {}
        stats = {"ok": 0, "failed": 0, "low_quality": 0, "short_circuited": 0}
        failure_reasons: Counter[str] = Counter()

        async def _one(url: str) -> tuple[str, Optional[Record], Optional[str]]:
            domain = self._domain_key_for_url(url)
            domain_sem = domain_semaphores.get(domain)
            if domain_sem is None:
                domain_sem = asyncio.Semaphore(self._scrapling_primary_per_domain_concurrency)
                domain_semaphores[domain] = domain_sem
            try:
                async with semaphore:
                    async with domain_sem:
                        delay_sec = self._sample_scrapling_primary_delay()
                        if delay_sec > 0:
                            await asyncio.sleep(delay_sec)
                        record, reason = await self._try_scrapling(
                            url,
                            timeout_sec=self._scrapling_primary_timeout_sec,
                            retries=self._scrapling_primary_retries,
                        )
            except Exception as exc:
                reason = f"scrapling_primary_exception:{type(exc).__name__}:{exc}"
                logger.bind(trace=True).warning(f"scrapling 主抓取异常: url={url} reason={reason}")
                return url, None, self._sanitize_failure_reason(reason)
            return url, record, self._sanitize_failure_reason(reason)

        for chunk_start in range(0, len(urls), self._scrapling_primary_chunk_size):
            chunk_urls = urls[chunk_start:chunk_start + self._scrapling_primary_chunk_size]
            tasks = [asyncio.create_task(_one(url)) for url in chunk_urls]
            for task in asyncio.as_completed(tasks):
                url, record, reason = await task

                if record is None:
                    stats["failed"] += 1
                    clean_reason = reason or "scrapling_primary_failed"
                    failure_reasons[clean_reason] += 1
                    self._mark_failure(url, clean_reason)
                    yield None, url
                    continue

                content_len = self._record_content_chars(record)
                if content_len < self._scrapling_quality_min_chars:
                    stats["low_quality"] += 1
                    reason_key = f"scrapling_primary_low_quality:min={self._scrapling_quality_min_chars}"
                    failure_reasons[reason_key] += 1
                    self._mark_failure(
                        url,
                        f"scrapling_primary_low_quality:content_chars={content_len},min={self._scrapling_quality_min_chars}",
                    )
                    logger.bind(trace=True).info(
                        f"scrapling 质量不足，转 rescue: url={url} "
                        f"content_chars={content_len} min={self._scrapling_quality_min_chars}"
                    )
                    yield None, url
                    continue

                self._append_output(record)
                stats["ok"] += 1
                yield record, None

            processed = stats["ok"] + stats["failed"] + stats["low_quality"]
            degraded = stats["failed"] + stats["low_quality"]
            if (
                processed >= self._scrapling_short_circuit_sample_size
                and processed > 0
                and (degraded / processed) >= self._scrapling_short_circuit_fail_ratio
                and (chunk_start + self._scrapling_primary_chunk_size) < len(urls)
            ):
                remaining_urls = urls[chunk_start + self._scrapling_primary_chunk_size:]
                stats["short_circuited"] += len(remaining_urls)
                logger.warning(
                    "scrapling 主抓取成功率过低，提前短路到 rescue: "
                    f"processed={processed} degraded={degraded} "
                    f"ratio={degraded / processed:.2f} remaining={len(remaining_urls)}"
                )
                for remaining_url in remaining_urls:
                    yield None, remaining_url
                break

        logger.debug(
            f"scrapling 主抓取统计: ok={stats['ok']} "
            f"low_quality={stats['low_quality']} failed={stats['failed']} "
            f"short_circuited={stats['short_circuited']}"
        )
        if failure_reasons:
            summary = "; ".join(f"{reason} x{count}" for reason, count in failure_reasons.most_common(5))
            logger.info(f"scrapling 主抓取失败原因TOP: {summary}")

    async def _crawl_with_crawl4ai_stream(
        self,
        urls: List[str],
        *,
        pre_succeeded_urls: Optional[set[str]] = None,
    ) -> AsyncIterator[Record]:
        if not urls:
            return

        succeeded_urls: set[str] = set(pre_succeeded_urls or set())
        failed_urls: List[str] = []

        proxy_strategy = self._build_proxy_strategy()
        target_elements = self._infer_single_domain_targets(urls)
        excluded_selectors = self._infer_single_domain_excluded(urls)
        run_config = self._build_run_config(
            proxy_strategy,
            target_elements=target_elements,
            excluded_selectors=excluded_selectors,
        )
        browser_config = self._build_browser_config()
        dispatcher = MemoryAdaptiveDispatcher(
            memory_threshold_percent=90.0,
            check_interval=1.0,
            max_session_permit=self.config.concurrent_crawls,
        )

        async with self._init_crawler(browser_config, use_undetected=True) as crawler:
            stream = await crawler.arun_many(urls=urls, config=run_config, dispatcher=dispatcher)
            async for result in stream:
                record, retry_url, failure_reason = self._process_result(result)
                if record is None:
                    if retry_url:
                        failed_urls.append(retry_url)
                        self._mark_failure(retry_url, failure_reason or "crawl4ai_failed")
                    continue
                if record.url in succeeded_urls:
                    continue
                succeeded_urls.add(record.url)
                self._append_output(record)
                yield record

        fallback_records = await self._crawl_failed_urls_with_fallback(
            failed_urls,
            succeeded_urls=succeeded_urls,
        )
        for record in fallback_records:
            if record.url in succeeded_urls:
                continue
            succeeded_urls.add(record.url)
            yield record

    @staticmethod
    def _record_content_chars(record: Record) -> int:
        content = record.content
        if isinstance(content, str):
            return len(content.strip())
        if content is None:
            return 0
        return len(str(content).strip())

    # ------------------------------------------------------------------
    # Crawl4AI 配置封装
    # ------------------------------------------------------------------
    def _build_proxy_strategy(self) -> Optional[RoundRobinProxyStrategy]:
        proxies_env = os.getenv("PROXIES")
        if not proxies_env:
            return None
        proxies = ProxyConfig.from_env("PROXIES")
        return RoundRobinProxyStrategy(proxies)

    def _build_run_config(
        self,
        proxy_strategy: Optional[RoundRobinProxyStrategy],
        *,
        target_elements: Optional[List[str]] = None,
        excluded_selectors: Optional[List[str]] = None,
    ) -> CrawlerRunConfig:
        mean_delay, max_range = self._build_delay_window()
        return CrawlerRunConfig(
            cache_mode=CacheMode.DISABLED,
            stream=True,
            verbose=False,
            log_console=False,
            proxy_rotation_strategy=proxy_strategy,
            markdown_generator=DefaultMarkdownGenerator(
                content_filter=PruningContentFilter(threshold=0.48, threshold_type="dynamic"),
                options={"ignore_links": True, "ignore_images": True},
            ),
            only_text=True,
            exclude_external_links=True,
            exclude_internal_links=True,
            exclude_social_media_links=True,
            exclude_all_images=True,
            magic=True,
            simulate_user=True,
            override_navigator=True,
            mean_delay=mean_delay,
            max_range=max_range,
            page_timeout=int(self.config.browser_timeout * 1000),
            # 如果提供了更精确的 main content selectors，则只在这些容器内生成 markdown
            target_elements=target_elements,
            # 显式排除的容器（如导航栏、推荐区等）
            # 注意：excluded_selector 需要是字符串，如果有多个选择器需要用逗号分隔
            excluded_selector=self._convert_excluded_selectors_to_string(excluded_selectors),
        )

    def _build_browser_config(self) -> BrowserConfig:
        headers = {
            "Accept-Language": "en-US,en;q=0.9",
            "DNT": "1",
            "Upgrade-Insecure-Requests": "1",
        }
        # 默认采用 undetected/stealth 配置，尽量避免简单反爬
        return BrowserConfig(
            enable_stealth=True,
            headless=True,
            verbose=False,
            user_agent_mode="random",
            headers=headers,
        )

    def _init_crawler(self, browser_config: BrowserConfig, use_undetected: bool = True) -> AsyncWebCrawler:
        """初始化 AsyncWebCrawler。

        - 默认使用 UndetectedAdapter + AsyncPlaywrightCrawlerStrategy；
        - 如果未来需要精简/禁用 undetected，可以将 `use_undetected` 设为 False，
          直接使用默认的 AsyncWebCrawler(config=browser_config)。
        """
        quiet_logger = AsyncLogger(verbose=False)
        if not use_undetected:
            return AsyncWebCrawler(config=browser_config, logger=quiet_logger)

        adapter = UndetectedAdapter()
        strategy = AsyncPlaywrightCrawlerStrategy(
            browser_config=browser_config,
            browser_adapter=adapter,
            logger=quiet_logger,
        )
        return AsyncWebCrawler(crawler_strategy=strategy, config=browser_config, logger=quiet_logger)

    def _build_delay_window(self) -> tuple[float, float]:
        minimum = max(0.0, float(self.config.request_delay_min))
        maximum = max(minimum, float(self.config.request_delay_max))
        mean_delay = (minimum + maximum) / 2.0
        max_range = max(0.0, maximum - minimum)
        return mean_delay, max_range

    def _convert_excluded_selectors_to_string(self, excluded_selectors: Optional[List[str]]) -> Optional[str]:
        """将排除选择器列表转换为逗号分隔的字符串，符合 Crawl4AI 的 excluded_selector 参数要求。"""
        if not excluded_selectors:
            return None
        # 过滤空字符串并去除前后空格
        clean_selectors = [s.strip() for s in excluded_selectors if s and s.strip()]
        if not clean_selectors:
            return None
        return ", ".join(clean_selectors)

    # ------------------------------------------------------------------
    # main content selectors 配置加载 & 域名推断
    # ------------------------------------------------------------------
    def _load_domain_selectors(self) -> tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        """加载域名 -> main content CSS selectors 映射。

        优先读取 `config/main_content_selectors.py`（支持默认值 + 继承），
        兼容旧的 `config/main_content_selectors.json`。
        """
        py_path = Path("config/main_content_selectors.py")
        json_path = Path("config/main_content_selectors.json")

        data: Any = None
        default_targets: List[str] = []
        default_excludes: List[str] = []

        if py_path.exists():
            try:
                import importlib.util

                spec = importlib.util.spec_from_file_location("main_content_selectors", py_path)
                if not spec or not spec.loader:
                    raise RuntimeError("无法加载 main_content_selectors.py")
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)  # type: ignore[call-arg]
                data = getattr(module, "DOMAIN_SELECTORS", None)
                default_targets = list(getattr(module, "DEFAULT_TARGETS", []) or [])
                default_excludes = list(getattr(module, "DEFAULT_EXCLUDE", []) or [])
            except Exception as exc:
                logger.warning(f"加载 main_content_selectors.py 失败: {exc}")
                data = None

        if data is None and json_path.exists():
            try:
                with json_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as exc:
                logger.warning(f"加载 main_content_selectors.json 失败: {exc}")
                return {}, {}

        if not isinstance(data, dict):
            return {}, {}

        targets: Dict[str, List[str]] = {}
        excludes: Dict[str, List[str]] = {}

        for domain, value in data.items():
            if domain.startswith("_"):
                # 跳过 _comment 等元信息
                continue

            dom_key = domain.lower()

            # 旧格式：直接是 selector list
            if isinstance(value, list):
                clean = [str(s) for s in value if isinstance(s, str) and s.strip()]
                if clean:
                    targets[dom_key] = [*default_targets, *clean] if default_targets else clean
                continue

            # 新格式：{ "targets": [...], "exclude": [...], "override_targets": bool, "override_exclude": bool }
            if not isinstance(value, dict):
                continue

            raw_targets = value.get("targets") or []
            raw_excludes = value.get("exclude") or []
            override_targets = bool(value.get("override_targets", False))
            override_excludes = bool(value.get("override_exclude", False))

            t_clean = [str(s) for s in raw_targets if isinstance(s, str) and s.strip()]
            e_clean = [str(s) for s in raw_excludes if isinstance(s, str) and s.strip()]

            final_targets = t_clean if override_targets else [*default_targets, *t_clean]
            final_excludes = e_clean if override_excludes else [*default_excludes, *e_clean]

            if final_targets:
                targets[dom_key] = final_targets
            if final_excludes:
                excludes[dom_key] = final_excludes

        return targets, excludes

    def _infer_single_domain_targets(self, urls: List[str]) -> Optional[List[str]]:
        """如果本次爬取只涉及单一主机，并且在配置中定义了 targets，则返回之。

        多域场景或未配置时返回 None，退回默认的全页抓取行为。
        """
        domains = set()
        for u in urls:
            try:
                netloc = urlparse(u).netloc or ""
            except Exception:
                continue
            host = netloc.split("@")[-1].split(":")[0]
            if host:
                domains.add(host.lower())
        if len(domains) != 1:
            return None
        (only_domain,) = tuple(domains)

        # 精确匹配 host，如果你希望合并子域，可以自己在配置里只写主域并按需扩展这里的逻辑
        return self._domain_target_selectors.get(only_domain)

    def _infer_single_domain_excluded(self, urls: List[str]) -> Optional[List[str]]:
        """如果本次爬取只涉及单一主机，并且在配置中定义了 exclude，则返回之。"""
        domains = set()
        for u in urls:
            try:
                netloc = urlparse(u).netloc or ""
            except Exception:
                continue
            host = netloc.split("@")[-1].split(":")[0]
            if host:
                domains.add(host.lower())
        if len(domains) != 1:
            return None
        (only_domain,) = tuple(domains)
        return self._domain_excluded_selectors.get(only_domain)

    # ------------------------------------------------------------------
    # 结果处理 & 序列化
    # ------------------------------------------------------------------
    def _process_result(self, result: Any) -> tuple[Optional[Record], Optional[str], Optional[str]]:
        """将 Crawl4AI 的结果转换为 `Record`。

        - 失败的请求只记录日志，不会输出 `Record`；
        - 不再尝试从 HTML 中抽取发布时间（pubtime），
          `published_at` 字段使用空字符串占位，后续可以由 URL 池/上游填充。
        """
        url = str(getattr(result, "input_url", "") or getattr(result, "url", "")).strip()
        success = bool(getattr(result, "success", False))
        error_message = getattr(result, "error_message", None) or getattr(result, "error", None)

        if not success:
            reason = self._sanitize_failure_reason(f"crawl4ai_error:{error_message or 'unknown'}")
            logger.bind(trace=True).warning(f"爬取失败: url={url} reason={reason}")
            return None, url or None, reason

        cleaned_html = getattr(result, "cleaned_html", None) or ""
        raw_html = getattr(result, "html", None) or ""
        html = cleaned_html or raw_html or ""

        markdown_text_raw = self._normalize_fit_markdown(getattr(result, "markdown", None))
        record = self._build_record(
            url=url,
            html=html,
            markdown_text=markdown_text_raw,
            allow_markdown_passthrough=False,
        )
        if record is None:
            reason = "crawl4ai_empty_record"
            logger.bind(trace=True).warning(f"爬取结果为空，加入 fallback 队列: url={url}")
            return None, url or None, reason
        return record, None, None

    def _build_record(
        self,
        *,
        url: str,
        html: str,
        markdown_text: str,
        allow_markdown_passthrough: bool,
    ) -> Optional[Record]:
        extractor = Extractor(html)
        cleaned_markdown = extractor.extract_content(markdown_text)

        content = cleaned_markdown.strip()
        if not content and html:
            content = self._extract_content_from_html(html, extractor).strip()
        if not content and allow_markdown_passthrough:
            content = (markdown_text or "").strip()

        if not html and not content:
            return None

        title = extractor.extract_title() or self._derive_title_from_content(content)
        description = extractor.extract_description() or self._derive_description_from_content(content)
        meta = self._url_meta.get(url, {})
        published_at = meta.get("published_at", "")

        return Record(
            id=str(uuid.uuid4()),
            source=self._infer_source(url),
            url=url,
            title=title,
            description=description,
            content=content,
            published_at=published_at,
            language="en",
        )

    def _extract_content_from_html(self, html: str, extractor: Extractor) -> str:
        if not html.strip() or trafilatura is None:
            return ""
        try:
            text = trafilatura.extract(
                html,
                output_format="markdown",
                include_tables=True,
                include_images=False,
                include_links=False,
            )
        except Exception as exc:
            logger.debug(f"trafilatura 提取失败: {exc}")
            return ""
        if not text:
            return ""
        cleaned = extractor.extract_content(text)
        if cleaned.strip():
            return cleaned
        return text.strip()

    def _derive_title_from_content(self, content: str) -> str:
        if not content:
            return ""
        for line in content.splitlines():
            stripped = line.strip()
            if stripped:
                return stripped[:200]
        return ""

    def _derive_description_from_content(self, content: str) -> Optional[str]:
        if not content:
            return None
        compact = re.sub(r"\s+", " ", content).strip()
        if not compact:
            return None
        return compact[:240]

    def _normalize_fit_markdown(self, markdown: Any) -> str:
        if not markdown:
            return ""
        if isinstance(markdown, str):
            return markdown
        fit_md = getattr(markdown, "fit_markdown", None)
        if isinstance(fit_md, str) and fit_md.strip():
            return fit_md
        raw_md = getattr(markdown, "raw_markdown", None)
        if isinstance(raw_md, str) and raw_md.strip():
            return raw_md
        return ""

    async def _crawl_failed_urls_with_fallback(self, failed_urls: List[str], *, succeeded_urls: set[str]) -> List[Record]:
        pending_urls = []
        seen = set(succeeded_urls)
        for url in failed_urls:
            u = str(url or "").strip()
            if not u or u in seen:
                continue
            seen.add(u)
            pending_urls.append(u)

        if not pending_urls:
            return []

        logger.debug(
            f"开始 fallback 爬取: pending={len(pending_urls)} "
            f"(scrapling={self._enable_scrapling_fallback}, jina={self._enable_jina_reader_fallback})"
        )

        timeout = httpx.Timeout(self._fallback_timeout_sec, connect=min(10.0, self._fallback_timeout_sec))
        semaphore = asyncio.Semaphore(self._fallback_concurrency)
        stats = {"scrapling": 0, "jina_reader": 0, "failed": 0}
        records: List[Record] = []

        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": "future-forecasting-jina-fallback/1.0"},
        ) as http_client:
            tasks = [
                asyncio.create_task(self._crawl_single_with_fallback(url, semaphore=semaphore, http_client=http_client))
                for url in pending_urls
            ]

            for task in asyncio.as_completed(tasks):
                try:
                    failed_url, record, stage, failure_reason = await task
                except Exception as exc:
                    logger.warning(f"fallback 任务异常: {exc}")
                    stats["failed"] += 1
                    continue

                stats[stage] = stats.get(stage, 0) + 1
                if record is None:
                    self._mark_failure(failed_url, failure_reason or "fallback_failed")
                    continue
                records.append(record)
                self._append_output(record)

        logger.debug(
            "fallback 完成: "
            f"scrapling={stats.get('scrapling', 0)} "
            f"jina_reader={stats.get('jina_reader', 0)} "
            f"failed={stats.get('failed', 0)}"
        )
        return records

    async def _crawl_single_with_fallback(
        self,
        url: str,
        *,
        semaphore: asyncio.Semaphore,
        http_client: httpx.AsyncClient,
    ) -> tuple[str, Optional[Record], str, Optional[str]]:
        async with semaphore:
            failure_parts: list[str] = []
            if self._enable_scrapling_fallback:
                record, scrapling_reason = await self._try_scrapling(url)
                if record is not None:
                    return url, record, "scrapling", None
                if scrapling_reason:
                    failure_parts.append(f"scrapling_fallback={scrapling_reason}")

            if self._enable_jina_reader_fallback:
                record, jina_reason = await self._try_jina_reader(url, http_client=http_client)
                if record is not None:
                    return url, record, "jina_reader", None
                if jina_reason:
                    failure_parts.append(f"jina_reader={jina_reason}")

        reason = "; ".join(failure_parts) if failure_parts else "fallback_failed"
        logger.bind(trace=True).warning(f"fallback 最终失败: url={url} reason={reason}")
        return url, None, "failed", self._sanitize_failure_reason(reason)

    async def _try_scrapling(
        self,
        url: str,
        *,
        timeout_sec: Optional[float] = None,
        retries: Optional[int] = None,
    ) -> tuple[Optional[Record], Optional[str]]:
        if AsyncFetcher is None:
            return None, "scrapling_unavailable"
        effective_timeout = float(timeout_sec if timeout_sec is not None else self._fallback_timeout_sec)
        effective_retries = int(retries if retries is not None else max(1, int(self.config.retry_attempts)))
        try:
            response = await AsyncFetcher.get(
                url,
                follow_redirects=True,
                timeout=effective_timeout,
                retries=max(1, effective_retries),
                stealthy_headers=True,
            )
        except Exception as exc:
            reason = self._sanitize_failure_reason(f"scrapling_exception:{type(exc).__name__}:{exc}")
            logger.bind(trace=True).warning(f"Scrapling 失败: url={url} reason={reason}")
            return None, reason

        status = int(getattr(response, "status", 0) or 0)
        if status >= 400:
            reason = f"scrapling_http_{status}"
            logger.bind(trace=True).warning(f"Scrapling HTTP 异常: url={url} status={status}")
            return None, reason

        html = self._extract_scrapling_html(response)
        text = self._extract_scrapling_text(response)
        record = self._build_record(
            url=url,
            html=html,
            markdown_text=text,
            allow_markdown_passthrough=True,
        )
        if record is None:
            return None, "scrapling_empty_record"
        return record, None

    async def _try_jina_reader(
        self,
        url: str,
        *,
        http_client: httpx.AsyncClient,
    ) -> tuple[Optional[Record], Optional[str]]:
        reader_url = self._jina_reader_prefix + url
        try:
            response = await http_client.get(reader_url)
        except Exception as exc:
            reason = self._sanitize_failure_reason(f"jina_reader_exception:{type(exc).__name__}:{exc}")
            logger.bind(trace=True).warning(f"Jina Reader 失败: url={url} reason={reason}")
            return None, reason

        if response.status_code >= 400:
            reason = f"jina_reader_http_{response.status_code}"
            logger.bind(trace=True).warning(f"Jina Reader HTTP 异常: url={url} status={response.status_code}")
            return None, reason

        text = self._clean_jina_reader_text(response.text)
        if not text:
            return None, "jina_reader_empty_text"
        record = self._build_record(
            url=url,
            html="",
            markdown_text=text,
            allow_markdown_passthrough=True,
        )
        if record is None:
            return None, "jina_reader_empty_record"
        return record, None

    @staticmethod
    def _extract_scrapling_html(response: Any) -> str:
        html = getattr(response, "html_content", None)
        if html is not None:
            text = str(html).strip()
            if text:
                return text
        body = getattr(response, "body", None)
        if isinstance(body, bytes):
            return body.decode("utf-8", errors="ignore").strip()
        if isinstance(body, str):
            return body.strip()
        return ""

    @staticmethod
    def _extract_scrapling_text(response: Any) -> str:
        try:
            all_text = response.get_all_text(separator="\n", strip=True)
        except Exception:
            return ""
        return str(all_text or "").strip()

    @staticmethod
    def _clean_jina_reader_text(text: str) -> str:
        raw = str(text or "").replace("\r\n", "\n").strip()
        if not raw:
            return ""
        marker = "Markdown Content:"
        if marker in raw:
            raw = raw.split(marker, 1)[1].strip()
        return raw

    def _filter_markdown_lines(self, markdown: str) -> str:
        """行级清洗：去掉明显太短的正文行，并裁掉末尾纯 heading 区域。"""
        if not markdown:
            return ""

        kept: list[str] = []
        for raw_line in markdown.splitlines():
            line = raw_line.rstrip()
            if not line:
                continue

            if "#" in line:
                kept.append(line)
                continue

            if len(line.strip()) >= 100:
                kept.append(line)

        if not kept:
            return ""

        # 去掉末尾连续的包含 '#' 的行（典型尾部导航）
        last_non_hash_idx = -1
        for i in range(len(kept) - 1, -1, -1):
            if "#" not in kept[i]:
                last_non_hash_idx = i
                break

        if last_non_hash_idx == -1:
            return ""

        kept = kept[: last_non_hash_idx + 1]
        return "\n".join(kept)

    def _append_raw_markdown_debug(self, url: str, markdown_raw: str) -> None:
        """将每条结果的 markdown 文本写入调试文件（当前为清洗后的正文）。"""
        if not markdown_raw:
            return
        try:
            self._raw_markdown_path.parent.mkdir(parents=True, exist_ok=True)
            with self._raw_markdown_path.open("a", encoding="utf-8") as f:
                f.write("URL: " + url + "\n")
                f.write(markdown_raw)
                f.write("\n" + "-" * 80 + "\n")
        except Exception as exc:
            logger.debug(f"写入 raw markdown 调试文件失败: {exc}")

    def _infer_source(self, url: str) -> str:
        try:
            netloc = urlparse(url).netloc or "unknown"
            # 去掉用户名、端口等，只保留域名部分
            host = netloc.split("@")[
                -1
            ]  # user:pass@host:port -> host:port
            host = host.split(":")[0]
            host = host or "unknown"
        except Exception:
            host = "unknown"
        return f"news/{host}"

    # ------------------------------------------------------------------
    # 输入 URL 加载 & 路径工具
    # ------------------------------------------------------------------
    def _load_urls(self, source_path: Optional[Path]) -> List[str]:
        if not source_path:
            return []
        if not source_path.exists():
            logger.error(f"URL 输入文件不存在: {source_path}")
            return []

        urls: List[str] = []
        self._url_meta.clear()

        with source_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        url = obj.get("url")
                        published_at = (
                            obj.get("published_at")
                            or obj.get("gkg_date")
                            or obj.get("created_at")
                            or obj.get("create_at")
                        )
                        if url:
                            self._url_meta[str(url)] = {
                                "published_at": str(published_at) if published_at else "",
                            }
                    else:
                        url = None
                except json.JSONDecodeError:
                    url = line

                if url:
                    urls.append(str(url))
        return urls

    def _normalize_urls_input(self, urls_input: List[Any]) -> List[str]:
        """Normalize mixed URL inputs (str or dict) and capture metadata."""
        urls: List[str] = []
        self._url_meta.clear()

        for item in urls_input:
            if isinstance(item, str):
                urls.append(item)
                continue
            if isinstance(item, dict):
                url = item.get("url")
                if not url:
                    continue
                published_at = (
                    item.get("published_at")
                    or item.get("gkg_date")
                    or item.get("created_at")
                    or item.get("create_at")
                )
                self._url_meta[str(url)] = {
                    "published_at": str(published_at) if published_at else "",
                }
                urls.append(str(url))

        return urls

    def _append_output(self, record: Record) -> None:
        """将记录追加到输出文件"""
        try:
            with self.output_path.open("a", encoding="utf-8") as f:
                f.write(record.to_json() + "\n")
        except Exception as exc:
            logger.error(f"写入输出文件失败: {exc}")

    def _default_output_path(self) -> Path:
        ts = time.strftime("%Y%m%d_%H%M%S")
        return self.config.processed_data_dir / f"crawl_results_{ts}.jsonl"
