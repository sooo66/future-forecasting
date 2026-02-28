"""简单可用的 Crawl4AI 爬虫实现

- 继承自 ``crawl4ai.hub.BaseCrawler``，对外暴露单一的 ``run`` 方法；
- 依赖 ``Config``，但支持直接传入 dict，方便在测试中构造临时配置；
- 输出与 ``utils.models.Record`` 对齐：每条爬取结果都序列化为一条 JSONL 记录。

当前目标是「能跑且易测」，未特别优化扩展性或高级反爬策略。
"""
from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

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

from info.crawler.proxy_pool import ProxyManager
from utils.config import Config
from utils.models import Record
from info.crawler.extractor import Extractor


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

        # 记录 URL -> 元信息（published_at/tags 等），便于透传到 Record
        self._url_meta: dict[str, dict[str, Any]] = {}

        self.proxy_manager = ProxyManager(
            proxy_file=proxy_file,
            use_proxy=self.config.use_proxy if use_proxy is None else use_proxy,
        )

        self._extractor = Extractor("")  # 先初始化为空，后续重新创建

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
        """运行爬虫并返回 `Record` 列表。

        - 如果在初始化时传入了 `urls`，则直接使用；
        - 否则从 `source_path` 加载 JSONL/纯文本 URL 列表。
        """
        if self.urls is not None:
            urls = self._normalize_urls_input(self.urls)
        else:
            urls = self._load_urls(self.source_path)
        if not urls:
            logger.warning("输入 URL 为空，跳过爬取")
            return []

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.proxy_manager.refresh_env()

        proxy_strategy = self._build_proxy_strategy()

        # 如果本次 URL 只来自单一域名，并且在配置中定义了 main content selectors，
        # 则将其传递给 Crawl4AI 的 target_elements / excluded_selector 以聚焦正文容器。
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

        records: List[Record] = []

        async with self._init_crawler(browser_config, use_undetected=True) as crawler:
            stream = await crawler.arun_many(urls=urls, config=run_config, dispatcher=dispatcher)
            async for result in stream:
                record = self._process_result(result)
                if record is None:
                    continue
                records.append(record)
                self._append_output(record)

        logger.info(f"爬取完成，总记录数: {len(records)}，输出: {self.output_path}")
        return records

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
            user_agent_mode="random",
            headers=headers,
        )

    def _init_crawler(self, browser_config: BrowserConfig, use_undetected: bool = True) -> AsyncWebCrawler:
        """初始化 AsyncWebCrawler。

        - 默认使用 UndetectedAdapter + AsyncPlaywrightCrawlerStrategy；
        - 如果未来需要精简/禁用 undetected，可以将 `use_undetected` 设为 False，
          直接使用默认的 AsyncWebCrawler(config=browser_config)。
        """
        if not use_undetected:
            return AsyncWebCrawler(config=browser_config)

        adapter = UndetectedAdapter()
        strategy = AsyncPlaywrightCrawlerStrategy(browser_config=browser_config, browser_adapter=adapter)
        return AsyncWebCrawler(crawler_strategy=strategy, config=browser_config)

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
    def _process_result(self, result: Any) -> Optional[Record]:
        """将 Crawl4AI 的结果转换为 `Record`。

        - 失败的请求只记录日志，不会输出 `Record`；
        - 不再尝试从 HTML 中抽取发布时间（pubtime），
          `published_at` 字段使用空字符串占位，后续可以由 URL 池/上游填充。
        """
        url = getattr(result, "url", "") or getattr(result, "input_url", "")
        success = bool(getattr(result, "success", False))
        error_message = getattr(result, "error_message", None) or getattr(result, "error", None)

        cleaned_html = getattr(result, "cleaned_html", None) or ""
        raw_html = getattr(result, "html", None) or ""
        html = cleaned_html or raw_html or ""

        if not success:
            logger.warning(f"爬取失败: {url} | {error_message}")
            return None

        markdown_text_raw = self._normalize_fit_markdown(getattr(result, "markdown", None))
        # 将原始 markdown 写入调试文件，方便排查重复等问题
        # self._append_raw_markdown_debug(url, markdown_text_raw)
        # 在 extractor 中做统一的 markdown 清洗
        cleaned_markdown = self._extractor.extract_content(markdown_text_raw)

        # self._append_raw_markdown_debug(url, cleaned_markdown)

        if not html and not cleaned_markdown:
            logger.warning(f"爬取结果为空: {url}")
            return None

        # 内容、标题、描述统一走 Extractor，逻辑在 tests/test_crawler.py 中有用例
        # 重新创建 extractor（需要 HTML）
        self._extractor = Extractor(html)
        
        title = self._extractor.extract_title() or ""
        description = self._extractor.extract_description()
        content = cleaned_markdown

        # URL 池已经针对英文做过筛选，这里直接标记为 "en"
        language = "en"
        meta = self._url_meta.get(url, {})
        tags = meta.get("tags")

        # 优先使用上游 URL 列表中携带的 created_at / create_at 作为发布时间
        published_at = meta.get("published_at", "")

        record = Record(
            id=str(uuid.uuid4()),
            source=self._infer_source(url),
            url=url,
            title=title,
            description=description,
            content=content,
            published_at=published_at,
            language=language,
            tags=tags,
        )
        return record

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
                            or obj.get("created_at")
                            or obj.get("create_at")
                            or obj.get("gkg_date")
                        )
                        tags = self._normalize_tags(obj.get("tags") or obj.get("themes"))
                        if url:
                            self._url_meta[str(url)] = {
                                "published_at": str(published_at) if published_at else "",
                                "tags": tags,
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
                    or item.get("created_at")
                    or item.get("create_at")
                    or item.get("gkg_date")
                )
                tags = self._normalize_tags(item.get("tags") or item.get("themes"))
                self._url_meta[str(url)] = {
                    "published_at": str(published_at) if published_at else "",
                    "tags": tags,
                }
                urls.append(str(url))

        return urls

    @staticmethod
    def _normalize_tags(value: Any) -> Optional[List[str]]:
        if not value:
            return None
        if isinstance(value, list):
            tags = [str(v).strip() for v in value if str(v).strip()]
            return tags or None
        if isinstance(value, str):
            sep = ";" if ";" in value else ","
            tags = [v.strip() for v in value.split(sep) if v.strip()]
            return tags or None
        return None

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
