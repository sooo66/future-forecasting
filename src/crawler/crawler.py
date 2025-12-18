"""Crawl4AI 爬虫模块"""
import asyncio
import json
import os
import random
import re
import hashlib
import time
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import uuid
from pathlib import Path
from loguru import logger

try:
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, LLMConfig
    from crawl4ai.content_filter_strategy import PruningContentFilter
    from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
    from crawl4ai.extraction_strategy import LLMExtractionStrategy
except ImportError:
    logger.error("Crawl4AI 未安装，请运行: pip install crawl4ai")
    raise
from pydantic import BaseModel, Field

from freeproxy import freeproxy

from utils.config import Config
from utils.models import Record
from url_pool_builder.builder import URLPoolBuilder
from .extractor import ContentExtractor


class ArticleLLMOutput(BaseModel):
    """用于 LLMExtractionStrategy 的结构化输出定义（只强制 content）"""
    
    title: Optional[str] = Field(None, description="Headline of the article (optional, keep original wording if present)")
    summary: Optional[str] = Field(None, description="Short a summary if meta description is absent")
    content: str = Field(..., description="Full article body text, verbatim, no paraphrasing or summarization")
    published_at: Optional[str] = Field(None, description="ISO8601 publish datetime if available, else null")
    language: Optional[str] = Field(None, description="Primary language code, e.g., en")
    tags: Optional[List[str]] = Field(None, description="Optional topical tags/keywords")


class NewsCrawler:
    """新闻爬虫
    
    使用 Crawl4AI 进行异步爬取，支持：
    - 并发控制（全局和每域名）
    - 随机延迟
    - User-Agent 轮换
    - 代理支持
    - 重试机制
    - 内容提取
    """
    
    def __init__(self, config: Config, url_pool_builder: URLPoolBuilder):
        self.config = config
        self.url_pool_builder = url_pool_builder
        self.extractor = ContentExtractor()
        
        # 配置参数
        self.concurrent_crawls = config.concurrent_crawls
        self.per_domain_concurrency = config.per_domain_concurrency
        self.retry_attempts = config.retry_attempts
        self.retry_delay_base = config.retry_delay_base
        self.request_delay_min = config.request_delay_min
        self.request_delay_max = config.request_delay_max
        self.user_agents = config.user_agents
        self.use_proxy = config.use_proxy
        self.use_browser = config.use_browser
        self.use_llm_extraction = config.use_llm_extraction
        self.llm_mode = config.llm_mode
        
        # 输出目录
        self.output_dir = config.processed_data_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 代理客户端
        self.proxy_client = None
        if self.use_proxy and freeproxy:
            try:
                # 尝试初始化代理客户端
                # 注意：freeproxy 的具体 API 可能因版本而异，这里提供一个通用实现
                proxy_sources = ["ProxylistProxiedSession"]
                for source_name in config.proxy_sources:
                    # 尝试动态获取代理源类
                    try:
                        if hasattr(freeproxy, source_name):
                            source_class = getattr(freeproxy, source_name)
                            proxy_sources.append(source_class)
                    except Exception:
                        logger.debug(f"无法加载代理源: {source_name}")
                
                if proxy_sources:
                    # 根据 freeproxy 的实际 API 调整
                    try:
                        init_proxied_session_cfg = {'filter_rule': {'country_code': ['US']}}
                        self.proxy_client = freeproxy.ProxiedSessionClient(proxy_sources=proxy_sources, init_proxied_session_cfg=init_proxied_session_cfg)
                        logger.info("代理客户端初始化成功")
                    except Exception as e:
                        logger.warning(f"代理客户端初始化失败（可能 API 不匹配）: {e}")
                        self.proxy_client = None
                else:
                    logger.warning("没有可用的代理源")
            except Exception as e:
                logger.warning(f"代理客户端初始化失败: {e}")
                self.proxy_client = None
        
        # 域名并发控制
        self.domain_semaphores: Dict[str, asyncio.Semaphore] = {}
        
        # 统计信息
        self.stats = {
            "total": 0,
            "success": 0,
            "failed": 0,
            "retries": 0
        }

        # LLM 提取策略（可选）
        self.llm_strategy = self._build_llm_strategy()

        # 精准定位问题用的 trace（默认开启，输出到 logs/crawl_trace_YYYY-MM-DD.jsonl）
        self._trace_enabled = os.getenv("FF_CRAWLER_TRACE", "1").lower() not in {"0", "false", "no", "off"}
        self._dump_artifacts = os.getenv("FF_CRAWLER_DUMP_ARTIFACTS", "0").lower() in {"1", "true", "yes", "on"}
        self._dump_on_failure = os.getenv("FF_CRAWLER_DUMP_ON_FAILURE", "1").lower() not in {"0", "false", "no", "off"}
        try:
            self._max_artifact_chars = int(os.getenv("FF_CRAWLER_MAX_ARTIFACT_CHARS", "200000"))
        except Exception:
            self._max_artifact_chars = 200000
        self._artifact_dir = self.config.log_dir / "crawl_artifacts"
        self._artifact_dir.mkdir(parents=True, exist_ok=True)

    def _trace(self, event: str, **fields):
        """仅用于定位链路问题的结构化日志（写入 crawl_trace*.jsonl）。"""
        if not self._trace_enabled:
            return
        logger.bind(trace=True, component="crawler", event=event, **fields).debug(event)

    def _sha256(self, text: Optional[str]) -> Optional[str]:
        if text is None:
            return None
        if not isinstance(text, str):
            text = str(text)
        return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()

    def _preview(self, text: Optional[str], limit: int = 800) -> Optional[str]:
        if text is None:
            return None
        if not isinstance(text, str):
            text = str(text)
        t = text.strip()
        if len(t) <= limit:
            return t
        return t[:limit] + f"...(truncated,{len(t)} chars)"

    def _text_stats(self, text: Optional[str]) -> Dict[str, int]:
        if text is None:
            return {"chars": 0, "words": 0}
        if not isinstance(text, str):
            text = str(text)
        if not text:
            return {"chars": 0, "words": 0}
        chars = len(text)
        words = len(re.findall(r"[A-Za-z]+", text))
        return {"chars": chars, "words": words}

    def _maybe_dump_artifact(self, *, url_id: str, kind: str, text: Optional[str], failure: bool = False) -> Optional[str]:
        """可选落盘保存大段文本，避免把全文塞进日志。"""
        if text is None:
            return None
        if not (self._dump_artifacts or (failure and self._dump_on_failure)):
            return None
        safe_kind = re.sub(r"[^a-zA-Z0-9_.-]+", "_", kind)[:80]
        path = self._artifact_dir / f"{url_id}_{safe_kind}.txt"
        payload = text[: self._max_artifact_chars]
        try:
            path.write_text(payload, encoding="utf-8", errors="replace")
            return str(path)
        except Exception:
            return None
    
    def _get_proxy(self) -> Optional[str]:
        """获取随机代理"""
        if not self.proxy_client:
            return None
        
        try:
            # 根据 freeproxy 的实际 API 调整
            # 可能的调用方式：
            # 1. proxy_client.getrandomproxy()
            # 2. proxy_client.get_proxy()
            # 3. 其他方式
            if hasattr(self.proxy_client, 'getrandomproxy'):
                proxy = self.proxy_client.getrandomproxy()
            elif hasattr(self.proxy_client, 'get_proxy'):
                proxy = self.proxy_client.get_proxy()
            else:
                # 如果 API 不匹配，返回 None
                logger.debug("代理客户端 API 不匹配，跳过代理")
                return None
            
            # 确保返回的代理格式正确（如 "http://ip:port"）
            if proxy and isinstance(proxy, str):
                return proxy
            return None
        except Exception as e:
            logger.debug(f"获取代理失败: {e}")
            return None
    
    def _get_random_user_agent(self) -> str:
        """获取随机 User-Agent"""
        return random.choice(self.user_agents) if self.user_agents else ""
    
    def _get_domain_semaphore(self, domain: str) -> asyncio.Semaphore:
        """获取域名的信号量（用于控制每域名并发）"""
        if domain not in self.domain_semaphores:
            self.domain_semaphores[domain] = asyncio.Semaphore(self.per_domain_concurrency)
        return self.domain_semaphores[domain]

    def _build_llm_strategy(self) -> Optional[LLMExtractionStrategy]:
        """初始化 LLMExtractionStrategy（可选）"""
        if not self.use_llm_extraction:
            return None
        
        api_token = os.getenv(self.config.llm_api_key_env)
        if not api_token and not self.config.llm_provider.startswith("ollama"):
            logger.warning(f"未找到环境变量 {self.config.llm_api_key_env}，LLM 提取已禁用")
            return None
        
        try:
            llm_config = LLMConfig(
                provider=self.config.llm_provider,
                api_token=api_token,
            )
            
            strategy = LLMExtractionStrategy(
                llm_config=llm_config,
                schema=ArticleLLMOutput.model_json_schema(),
                extraction_type="schema",
                instruction=self.config.llm_instruction,
                chunk_token_threshold=self.config.llm_chunk_token_threshold,
                overlap_rate=self.config.llm_overlap_rate,
                apply_chunking=self.config.llm_apply_chunking,
                input_format=self.config.llm_input_format,
                extra_args={
                    "temperature": self.config.llm_temperature,
                    "max_tokens": self.config.llm_max_output_tokens,
                },
                verbose=False,
            )
            
            logger.info(f"已启用 LLM 提取，provider={self.config.llm_provider}, mode={self.llm_mode}")
            return strategy
        except Exception as e:
            logger.warning(f"初始化 LLMExtractionStrategy 失败，回退到规则提取: {e}")
            return None
    
    async def _random_delay(self):
        """随机延迟"""
        delay = random.uniform(self.request_delay_min, self.request_delay_max)
        await asyncio.sleep(delay)
    
    async def _try_archive_is(self, url_id: str, original_url: str, browser_config, run_config) -> Optional[Dict]:
        """尝试使用 archive.is 获取存档内容以绕过 paywall
        
        archive.is 的工作方式：
        1. 访问 https://archive.is/{url} 会显示存档列表
        2. 最新的存档链接通常在页面中，格式类似 /newest/{timestamp} 或直接是完整URL
        3. 访问存档链接后，页面会重定向或显示存档内容
        """
        try:
            # 构建 archive.is 查询URL
            archive_query_url = f"https://archive.is/{original_url}"
            self._trace(
                "archive_query_start",
                url_id=url_id,
                url=original_url,
                archive_query_url=archive_query_url,
                has_run_config=bool(run_config),
            )
            
            async with AsyncWebCrawler(config=browser_config) as crawler:
                # 先访问查询页面
                if run_config:
                    query_result = await crawler.arun(url=archive_query_url, config=run_config)
                else:
                    query_result = await crawler.arun(url=archive_query_url)
            
                if not query_result.success or not query_result.html:
                    self._trace(
                        "archive_query_failed",
                        url_id=url_id,
                        url=original_url,
                        archive_query_url=archive_query_url,
                        success=bool(getattr(query_result, "success", False)),
                        error_message=getattr(query_result, "error_message", None),
                        html_stats=self._text_stats(getattr(query_result, "html", None)),
                    )
                    logger.warning(f"archive_fetch_failed url={original_url} reason=query_page_fetch_failed")
                    return None
                
                # 解析 archive.is 页面，查找存档链接
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(query_result.html, 'html.parser')
                
                archive_link = None
                archive_link_method = None

                # 优先根据 archive.is 的表格定位：//*[@id="row0"]/div[2]/a[1]
                row0 = soup.find(id="row0")
                if row0:
                    divs = row0.find_all("div")
                    if len(divs) >= 2:
                        anchors = divs[1].find_all("a", href=True)
                        if anchors:
                            href = anchors[0].get("href", "")
                            if href.startswith("/"):
                                archive_link = f"https://archive.is{href}"
                                archive_link_method = "row0_first_link"
                            elif href.startswith("http"):
                                archive_link = href
                                archive_link_method = "row0_first_link"
                
                # 备选方案：查找包含 /newest/ 或 /web/ 的链接
                if not archive_link:
                    for link in soup.find_all('a', href=True):
                        href = link.get('href', '')
                        if '/newest/' in href or '/web/' in href:
                            if href.startswith('/'):
                                archive_link = f"https://archive.is{href}"
                                archive_link_method = "newest_or_web_link"
                            elif href.startswith('http'):
                                archive_link = href
                                archive_link_method = "newest_or_web_link"
                            break
                
                # 再次备选：查找包含原始 URL 的存档链接
                if not archive_link:
                    for link in soup.find_all('a', href=True):
                        href = link.get('href', '')
                        if original_url in href and 'archive.is' in href:
                            archive_link = href
                            archive_link_method = "contains_original_url"
                            break
                
                # 最后兜底：当前页面足够长则直接尝试
                if not archive_link:
                    page_text = soup.get_text()
                    if len(page_text) > 1000:
                        archive_link = archive_query_url
                        archive_link_method = "fallback_use_query_page"
                
                if not archive_link:
                    self._trace(
                        "archive_link_not_found",
                        url_id=url_id,
                        url=original_url,
                        archive_query_url=archive_query_url,
                        query_html_stats=self._text_stats(query_result.html),
                    )
                    logger.warning(f"archive_link_not_found url={original_url}")
                    return None

                self._trace(
                    "archive_link_selected",
                    url_id=url_id,
                    url=original_url,
                    archive_query_url=archive_query_url,
                    archive_link=archive_link,
                    method=archive_link_method,
                )
                
                # 访问存档页面
                if run_config:
                    archive_result = await crawler.arun(url=archive_link, config=run_config)
                else:
                    archive_result = await crawler.arun(url=archive_link)
                
                if not archive_result.success or not archive_result.html:
                    self._trace(
                        "archive_content_fetch_failed",
                        url_id=url_id,
                        url=original_url,
                        archive_link=archive_link,
                        success=bool(getattr(archive_result, "success", False)),
                        error_message=getattr(archive_result, "error_message", None),
                        html_stats=self._text_stats(getattr(archive_result, "html", None)),
                    )
                    logger.warning(f"archive_content_fetch_failed url={original_url}")
                    return None

                # 处理 archive.is 的 markdown
                archive_markdown = None
                if hasattr(archive_result, 'markdown'):
                    if isinstance(archive_result.markdown, str):
                        archive_markdown = archive_result.markdown
                    elif hasattr(archive_result.markdown, 'markdown'):
                        archive_markdown = archive_result.markdown.markdown
                    elif hasattr(archive_result.markdown, 'raw_markdown'):
                        archive_markdown = archive_result.markdown.raw_markdown
                    else:
                        archive_markdown = str(archive_result.markdown)
                
                # 提取内容
                extracted = self.extractor.extract(
                    html=archive_result.html,
                    markdown=archive_markdown,
                    url=original_url,
                    source_name="",
                    gkg_date=None,
                    themes=None
                )

                content_stats = self._text_stats(extracted.get("content"))
                self._trace(
                    "archive_extracted",
                    url_id=url_id,
                    url=original_url,
                    archive_link=archive_link,
                    html_stats=self._text_stats(archive_result.html),
                    markdown_stats=self._text_stats(archive_markdown),
                    content_stats=content_stats,
                    title_preview=self._preview(extracted.get("title"), 200),
                )
                
                if extracted["content"] and len(extracted["content"].strip()) > 100:
                    logger.info(f"archive_success url={original_url} via={archive_link}")
                    return {
                        "content": extracted["content"],
                        "title": extracted["title"],
                        "summary": extracted["summary"]
                    }
                else:
                    html_path = self._maybe_dump_artifact(
                        url_id=url_id,
                        kind="archive_html",
                        text=archive_result.html,
                        failure=True,
                    )
                    md_path = self._maybe_dump_artifact(
                        url_id=url_id,
                        kind="archive_markdown",
                        text=archive_markdown,
                        failure=True,
                    )
                    self._trace(
                        "archive_content_too_short",
                        url_id=url_id,
                        url=original_url,
                        archive_link=archive_link,
                        content_stats=content_stats,
                        content_sha256=self._sha256(extracted.get("content")),
                        content_preview=self._preview(extracted.get("content")),
                        archive_html_path=html_path,
                        archive_markdown_path=md_path,
                    )
                    logger.warning(f"archive_content_too_short url={original_url} via={archive_link}")
        
        except Exception as e:
            self._trace(
                "archive_exception",
                url_id=url_id,
                url=original_url,
                error_type=type(e).__name__,
                error=str(e),
            )
            logger.warning(f"archive_exception url={original_url} err={type(e).__name__}:{e}")
        
        return None

    async def _run_llm_extraction(self, url_id: str, url: str, base_run_config_kwargs: Dict, browser_config) -> Optional[Dict]:
        """在需要时触发 LLM 提取（重用同一套反爬配置）"""
        if not self.llm_strategy:
            return None
        
        llm_run_kwargs = dict(base_run_config_kwargs)
        llm_run_kwargs["extraction_strategy"] = self.llm_strategy
        run_config_llm = CrawlerRunConfig(**llm_run_kwargs)
        
        try:
            t0 = time.monotonic()
            self._trace(
                "llm_fallback_start",
                url_id=url_id,
                url=url,
                llm_provider=self.config.llm_provider,
                llm_input_format=self.config.llm_input_format,
                chunk_token_threshold=self.config.llm_chunk_token_threshold,
                apply_chunking=self.config.llm_apply_chunking,
            )
            async with AsyncWebCrawler(config=browser_config) as crawler:
                llm_result = await crawler.arun(url=url, config=run_config_llm)
            dt_ms = int((time.monotonic() - t0) * 1000)

            llm_html = getattr(llm_result, "html", None)
            llm_md = None
            if hasattr(llm_result, "markdown"):
                if isinstance(llm_result.markdown, str):
                    llm_md = llm_result.markdown
                elif hasattr(llm_result.markdown, "markdown"):
                    llm_md = llm_result.markdown.markdown
                elif hasattr(llm_result.markdown, "raw_markdown"):
                    llm_md = llm_result.markdown.raw_markdown
                else:
                    llm_md = str(llm_result.markdown)
            llm_raw = getattr(llm_result, "extracted_content", None)

            llm_html_path = self._maybe_dump_artifact(url_id=url_id, kind="llm_fallback_html", text=llm_html, failure=False)
            llm_md_path = self._maybe_dump_artifact(url_id=url_id, kind="llm_fallback_markdown", text=llm_md, failure=False)
            llm_raw_path = self._maybe_dump_artifact(
                url_id=url_id,
                kind="llm_fallback_output_raw",
                text=str(llm_raw) if llm_raw is not None else None,
                failure=False,
            )
            self._trace(
                "llm_fallback_crawl_done",
                url_id=url_id,
                url=url,
                duration_ms=dt_ms,
                success=bool(getattr(llm_result, "success", False)),
                error_message=getattr(llm_result, "error_message", None),
                html_stats=self._text_stats(llm_html),
                markdown_stats=self._text_stats(llm_md),
                llm_raw_stats=self._text_stats(llm_raw),
                llm_raw_sha256=self._sha256(llm_raw),
                llm_raw_preview=self._preview(llm_raw),
                llm_html_path=llm_html_path,
                llm_markdown_path=llm_md_path,
                llm_output_path=llm_raw_path,
            )
            
            if llm_result.success and getattr(llm_result, "extracted_content", None):
                parsed = self._parse_llm_content(llm_result.extracted_content)
                if parsed:
                    content_stats = self._text_stats(parsed.get("content"))
                    self._trace(
                        "llm_fallback_parsed",
                        url_id=url_id,
                        url=url,
                        content_stats=content_stats,
                        title_preview=self._preview(parsed.get("title"), 200),
                        summary_preview=self._preview(parsed.get("summary"), 200),
                    )
                    logger.info(f"LLM 提取成功: {url}")
                    return parsed
                else:
                    # 解析失败：为定位问题，按需落盘输入/输出
                    llm_html_fail_path = self._maybe_dump_artifact(url_id=url_id, kind="llm_fallback_html", text=llm_html, failure=True)
                    llm_md_fail_path = self._maybe_dump_artifact(url_id=url_id, kind="llm_fallback_markdown", text=llm_md, failure=True)
                    llm_raw_fail_path = self._maybe_dump_artifact(
                        url_id=url_id,
                        kind="llm_fallback_output_raw",
                        text=str(llm_raw) if llm_raw is not None else None,
                        failure=True,
                    )
                    self._trace(
                        "llm_fallback_parse_failed",
                        url_id=url_id,
                        url=url,
                        llm_raw_sha256=self._sha256(llm_raw),
                        llm_raw_preview=self._preview(llm_raw),
                        llm_html_path=llm_html_fail_path,
                        llm_markdown_path=llm_md_fail_path,
                        llm_output_path=llm_raw_fail_path,
                    )
            else:
                # 爬取/抽取失败：按需落盘便于诊断
                self._maybe_dump_artifact(url_id=url_id, kind="llm_fallback_html", text=llm_html, failure=True)
                self._maybe_dump_artifact(url_id=url_id, kind="llm_fallback_markdown", text=llm_md, failure=True)
                self._maybe_dump_artifact(
                    url_id=url_id,
                    kind="llm_fallback_output_raw",
                    text=str(llm_raw) if llm_raw is not None else None,
                    failure=True,
                )
                logger.warning(f"LLM 提取失败: {url}")
        except Exception as e:
            self._trace(
                "llm_fallback_exception",
                url_id=url_id,
                url=url,
                error_type=type(e).__name__,
                error=str(e),
            )
            logger.warning(f"LLM 提取异常，忽略: {url} err={type(e).__name__}:{e}")
        
        return None

    def _parse_llm_content(self, content: str) -> Optional[Dict]:
        """解析 LLM 返回的 JSON 内容"""
        if not content:
            return None
        try:
            parsed = json.loads(content)
            if isinstance(parsed, list) and parsed:
                parsed = parsed[0]
            if isinstance(parsed, dict):
                return parsed
        except Exception as e:
            logger.debug(f"解析 LLM JSON 失败: {e}")
        return None

    def _is_proxy_error_content(self, text: Optional[str]) -> bool:
        """检测返回内容是否为代理错误提示，避免把错误页当正文"""
        if not text:
            return False
        indicators = [
            "using socks proxy", "socksio", "install httpx[socks]",
            "proxyerror", "proxy error", "proxy connection failed"
        ]
        lower = text.lower()
        return any(ind in lower for ind in indicators)
    
    async def _crawl_url(self, url_info: Dict) -> Optional[Record]:
        """爬取单个 URL"""
        url_id = url_info["id"]
        url = url_info["url"]
        domain = url_info["domain"]
        source_name = url_info.get("source_name", "")
        gkg_date = url_info.get("gkg_date")
        themes = url_info.get("themes")

        self._trace(
            "crawl_start",
            url_id=url_id,
            url=url,
            domain=domain,
            source_name=source_name,
            use_proxy=self.use_proxy,
            use_browser=self.use_browser,
            use_llm_extraction=bool(self.llm_strategy),
            llm_mode=self.llm_mode if self.llm_strategy else None,
            min_word_threshold=self.extractor.min_word_threshold,
        )
        
        # 获取域名信号量
        semaphore = self._get_domain_semaphore(domain)
        
        async with semaphore:
            # 随机延迟
            await self._random_delay()
            
            # 准备配置
            user_agent = self._get_random_user_agent()
            proxy = self._get_proxy() if self.use_proxy else None
            
            # 构建 source 字段
            source = f"news/{domain}"
            
            # 配置浏览器
            browser_config = None
            if self.use_browser:
                browser_config = BrowserConfig(
                    headless=self.config.browser_headless,
                    user_agent_mode="random"
                )
            
            # 配置内容过滤器 - 使用 PruningContentFilter 移除导航、侧边栏等
            # 根据 Crawl4AI 文档，PruningContentFilter 可以自动识别并移除非主要内容
            # 尝试配置排除选择器来移除更多无关元素
            try:
                # 尝试配置排除选择器（如果 API 支持）
                exclude_selectors = [
                    'nav', 'header', 'footer', 'aside', '.nav', '.navigation', 
                    '.navbar', '.menu', '.sidebar', '.advertisement', '.ad',
                    '.social', '.share', '.subscribe', '.newsletter', '.cookie',
                    '.breadcrumb', '.tags', '.related', '.comments', '.trending'
                ]
                
                pruning_kwargs = {
                    "threshold": 0.52,
                    "threshold_type": "fixed",
                    "min_word_threshold": self.extractor.min_word_threshold,
                    "exclude_selectors": exclude_selectors,
                }

                # 尝试使用增强的参数（如 threshold/min_word_threshold）
                try:
                    content_filter = PruningContentFilter(**pruning_kwargs)
                except TypeError:
                    # 如果当前版本不支持这些参数，降级为默认配置
                    logger.debug("PruningContentFilter 参数不兼容，使用默认配置")
                    content_filter = PruningContentFilter()
            except Exception as e:
                logger.debug(f"无法创建 PruningContentFilter: {e}")
                content_filter = None
            
            # 配置 Markdown 生成器 - 使用内容过滤器
            try:
                if content_filter:
                    markdown_generator = DefaultMarkdownGenerator(content_filter=content_filter)
                else:
                    markdown_generator = DefaultMarkdownGenerator()
            except Exception as e:
                logger.debug(f"无法创建 DefaultMarkdownGenerator: {e}")
                markdown_generator = None
            
            # 配置运行参数
            # 注意：Crawl4AI 的 RunConfig API 可能因版本而异
            # 这里提供一个兼容性实现
            run_config_kwargs = {}
            if user_agent:
                run_config_kwargs["user_agent"] = user_agent
            if proxy:
                run_config_kwargs["proxy"] = proxy
            if self.use_browser:
                run_config_kwargs["wait_for"] = "body"  # 等待 body 加载
                run_config_kwargs["page_timeout"] = self.config.browser_timeout * 1000
            
            # 设置 Markdown 生成器（如果可用）
            if markdown_generator:
                try:
                    run_config_kwargs["markdown_generator"] = markdown_generator
                except Exception as e:
                    logger.debug(f"无法设置 markdown_generator: {e}")
            
            # 尝试使用内容选择器来提取主要内容
            # 根据 Crawl4AI 文档，可以使用 content_selector 参数
            # 优先选择 article, main 等主要内容容器
            try:
                # 尝试设置内容选择器（如果 API 支持）
                if hasattr(CrawlerRunConfig, 'content_selector'):
                    run_config_kwargs["content_selector"] = "article, main, [role='article'], .article-content, .article-body, .post-content, .entry-content, .story-body, .content-body"
                elif hasattr(CrawlerRunConfig, 'css_selector'):
                    run_config_kwargs["css_selector"] = "article, main, [role='article'], .article-content, .article-body, .post-content, .entry-content, .story-body, .content-body"
            except Exception as e:
                logger.debug(f"无法设置内容选择器: {e}")
            
            base_run_config_kwargs = dict(run_config_kwargs)

            if self.llm_strategy and self.llm_mode == "always":
                run_config_kwargs["extraction_strategy"] = self.llm_strategy
            
            run_config = CrawlerRunConfig(**run_config_kwargs) if run_config_kwargs else None
            
            # 重试逻辑
            last_error = None
            current_proxy = proxy
            for attempt in range(self.retry_attempts):
                # 每次尝试根据当前代理重建 run_config
                if base_run_config_kwargs is not None:
                    attempt_kwargs = dict(base_run_config_kwargs)
                    if current_proxy:
                        attempt_kwargs["proxy"] = current_proxy
                    else:
                        attempt_kwargs.pop("proxy", None)
                    if self.llm_strategy and self.llm_mode == "always":
                        attempt_kwargs["extraction_strategy"] = self.llm_strategy
                    run_config = CrawlerRunConfig(**attempt_kwargs) if attempt_kwargs else None
                try:
                    attempt_no = attempt + 1
                    t0 = time.monotonic()
                    self._trace(
                        "fetch_attempt_start",
                        url_id=url_id,
                        url=url,
                        attempt=attempt_no,
                        proxy_mode="proxy" if current_proxy else "direct",
                        has_run_config=bool(run_config),
                        use_browser=self.use_browser,
                    )
                    async with AsyncWebCrawler(config=browser_config) as crawler:
                        # Crawl4AI 的 arun 方法参数可能因版本而异
                        if run_config:
                            result = await crawler.arun(url=url, config=run_config)
                        else:
                            # 如果没有 run_config，直接传递参数
                            result = await crawler.arun(
                                url=url,
                                user_agent=user_agent if user_agent else None,
                                proxy=current_proxy if current_proxy else None
                            )
                    dt_ms = int((time.monotonic() - t0) * 1000)
                        
                    if result.success:
                            # 代理错误页检测：如果返回的是代理错误提示，触发重试并改用直连
                            if self._is_proxy_error_content(result.html):
                                last_error = "proxy_error_content"
                                self._trace(
                                    "proxy_error_content",
                                    url_id=url_id,
                                    url=url,
                                    attempt=attempt_no,
                                    duration_ms=dt_ms,
                                    html_stats=self._text_stats(getattr(result, "html", None)),
                                )
                                logger.warning(f"proxy_error_content url={url} attempt={attempt_no} action=switch_to_direct")
                                current_proxy = None
                                continue

                            # 提取内容
                            html = result.html
                            
                            # 处理 markdown - Crawl4AI 的 markdown 可能是对象或字符串
                            # 根据文档，如果使用了 DefaultMarkdownGenerator，markdown 是一个对象
                            if hasattr(result, 'markdown'):
                                if isinstance(result.markdown, str):
                                    markdown = result.markdown
                                elif hasattr(result.markdown, 'markdown'):
                                    markdown = result.markdown.markdown
                                elif hasattr(result.markdown, 'raw_markdown'):
                                    markdown = result.markdown.raw_markdown
                                else:
                                    markdown = str(result.markdown)
                            else:
                                markdown = None

                            self._trace(
                                "fetch_attempt_success",
                                url_id=url_id,
                                url=url,
                                attempt=attempt_no,
                                duration_ms=dt_ms,
                                proxy_mode="proxy" if current_proxy else "direct",
                                html_stats=self._text_stats(html),
                                markdown_stats=self._text_stats(markdown),
                            )

                            extracted = self.extractor.extract(
                                html=html,
                                markdown=markdown,
                                url=url,
                                source_name=source_name,
                                gkg_date=gkg_date,
                                themes=themes
                            )

                            llm_payload = None
                            llm_raw = None
                            if self.llm_strategy and self.llm_mode == "always":
                                llm_raw = getattr(result, "extracted_content", None)
                                llm_payload = self._parse_llm_content(llm_raw)
                                html_path = self._maybe_dump_artifact(url_id=url_id, kind="fetch_html", text=html, failure=False)
                                md_path = self._maybe_dump_artifact(url_id=url_id, kind="fetch_markdown", text=markdown, failure=False)
                                llm_raw_path = self._maybe_dump_artifact(
                                    url_id=url_id,
                                    kind="llm_always_output_raw",
                                    text=str(llm_raw) if llm_raw is not None else None,
                                    failure=False,
                                )
                                self._trace(
                                    "llm_always_done",
                                    url_id=url_id,
                                    url=url,
                                    attempt=attempt_no,
                                    llm_provider=self.config.llm_provider,
                                    llm_input_format=self.config.llm_input_format,
                                    llm_raw_stats=self._text_stats(llm_raw),
                                    llm_raw_sha256=self._sha256(llm_raw),
                                    llm_raw_preview=self._preview(llm_raw),
                                    parsed_ok=bool(llm_payload),
                                    fetch_html_path=html_path,
                                    fetch_markdown_path=md_path,
                                    llm_output_path=llm_raw_path,
                                )
                                if llm_payload:
                                    self._trace(
                                        "llm_always_parsed",
                                        url_id=url_id,
                                        url=url,
                                        content_stats=self._text_stats(llm_payload.get("content")),
                                        title_preview=self._preview(llm_payload.get("title"), 200),
                                        summary_preview=self._preview(llm_payload.get("summary"), 200),
                                    )

                            content_stats = self._text_stats(extracted.get("content"))
                            self._trace(
                                "rule_extracted",
                                url_id=url_id,
                                url=url,
                                attempt=attempt_no,
                                title_preview=self._preview(extracted.get("title"), 200),
                                summary_preview=self._preview(extracted.get("summary"), 200),
                                content_stats=content_stats,
                            )
                            
                            # 检查内容是否为空（可能是paywall）
                            content = extracted["content"]
                            if (not content or len(content.strip()) < 100) and llm_payload is None:
                                # 尝试使用 archive.is 绕过 paywall
                                self._trace(
                                    "archive_trigger",
                                    url_id=url_id,
                                    url=url,
                                    reason="content_empty_or_short",
                                    content_stats=self._text_stats(content),
                                )
                                logger.info(f"content_empty_or_short_try_archive url={url}")
                                archive_result = await self._try_archive_is(url_id, url, browser_config, run_config)
                                
                                if archive_result and archive_result.get("content"):
                                    extracted["content"] = archive_result["content"]
                                    if not extracted["title"] or extracted["title"] == url:
                                        extracted["title"] = archive_result.get("title", extracted["title"])
                                    if not extracted["summary"]:
                                        extracted["summary"] = archive_result.get("summary", extracted["summary"])
                                    content = extracted["content"]
                                    self._trace(
                                        "archive_applied",
                                        url_id=url_id,
                                        url=url,
                                        content_stats=self._text_stats(content),
                                    )
                                    logger.info(f"archive_applied url={url}")

                            # LLM fallback: 在规则提取仍不足时触发
                            word_count = len(re.findall(r"[A-Za-z]+", content or ""))
                            if self.llm_strategy and self.llm_mode == "fallback" and word_count < self.extractor.min_word_threshold:
                                llm_payload = await self._run_llm_extraction(url_id, url, base_run_config_kwargs, browser_config)
                            
                            if llm_payload:
                                # 仅使用 LLM 提供的正文，避免任何摘要/改写被写入其它字段
                                content_from_llm = llm_payload.get("content") or ""
                                if isinstance(content_from_llm, list):
                                    content_from_llm = " ".join([str(x) for x in content_from_llm])
                                if content_from_llm:
                                    extracted["content"] = content_from_llm
                                    content = extracted["content"]
                                    word_count = len(re.findall(r"[A-Za-z]+", content or ""))
                                # 如果 summary 为空且 LLM 提供了摘要，则使用 LLM 摘要
                                if not extracted.get("summary"):
                                    llm_summary = llm_payload.get("summary")
                                    if isinstance(llm_summary, list):
                                        llm_summary = " ".join([str(x) for x in llm_summary])
                                    if llm_summary:
                                        extracted["summary"] = llm_summary

                                self._trace(
                                    "llm_payload_applied",
                                    url_id=url_id,
                                    url=url,
                                    content_stats=self._text_stats(extracted.get("content")),
                                    summary_preview=self._preview(extracted.get("summary"), 200),
                                )
                            
                            # 如果仍然没有摘要，使用正文首段作为兜底，避免为空
                            if not extracted.get("summary") and content:
                                paragraphs = content.split('\n\n')
                                if paragraphs:
                                    extracted["summary"] = paragraphs[0][:500]
                            
                            if not content or word_count < self.extractor.min_word_threshold:
                                last_error = f"content_too_short:words={word_count}"
                                content_path = self._maybe_dump_artifact(url_id=url_id, kind="final_content", text=content, failure=True)
                                fetch_html_path = self._maybe_dump_artifact(url_id=url_id, kind="fetch_html", text=html, failure=True)
                                fetch_markdown_path = self._maybe_dump_artifact(url_id=url_id, kind="fetch_markdown", text=markdown, failure=True)
                                llm_output_path = self._maybe_dump_artifact(
                                    url_id=url_id,
                                    kind="llm_output_raw_on_failure",
                                    text=str(llm_raw) if llm_raw is not None else None,
                                    failure=True,
                                )
                                llm_payload_path = self._maybe_dump_artifact(
                                    url_id=url_id,
                                    kind="llm_payload_on_failure",
                                    text=json.dumps(llm_payload, ensure_ascii=False) if isinstance(llm_payload, dict) else None,
                                    failure=True,
                                )
                                self._trace(
                                    "final_content_too_short",
                                    url_id=url_id,
                                    url=url,
                                    word_count=word_count,
                                    min_word_threshold=self.extractor.min_word_threshold,
                                    content_stats=self._text_stats(content),
                                    content_sha256=self._sha256(content),
                                    content_preview=self._preview(content),
                                    content_path=content_path,
                                    fetch_html_path=fetch_html_path,
                                    fetch_markdown_path=fetch_markdown_path,
                                    llm_output_path=llm_output_path,
                                    llm_payload_path=llm_payload_path,
                                )
                                logger.warning(f"content_too_short url={url} words={word_count} min_words={self.extractor.min_word_threshold}")
                                self.url_pool_builder.update_status(url_id, "failed", last_error)
                                self.stats["failed"] += 1
                                return None

                            # 创建 Record
                            record = Record(
                                id=str(uuid.uuid4()),
                                source=source,
                                url=url,
                                title=extracted["title"],
                                summary=extracted["summary"],
                                content=content,
                                published_at=extracted["published_at"],
                                language=extracted["language"],
                                tags=extracted["tags"]
                            )
                            
                            # 更新状态为成功
                            self.url_pool_builder.update_status(url_id, "success")
                            self.stats["success"] += 1

                            self._trace(
                                "final_success",
                                url_id=url_id,
                                url=url,
                                attempt=attempt_no,
                                content_stats=self._text_stats(content),
                                title_preview=self._preview(extracted.get("title"), 200),
                            )
                            
                            return record
                    else:
                            error_msg = result.error_message or 'unknown'
                            last_error = f"fetch_failed:{error_msg}"
                            self._trace(
                                "fetch_attempt_failed",
                                url_id=url_id,
                                url=url,
                                attempt=attempt_no,
                                duration_ms=dt_ms,
                                proxy_mode="proxy" if current_proxy else "direct",
                                error_message=error_msg,
                            )
                            logger.warning(f"fetch_failed url={url} attempt={attempt_no} proxy={'direct' if not current_proxy else 'proxy'} err={error_msg}")
                            
                except Exception as e:
                    last_error = str(e)
                    self._trace(
                        "fetch_attempt_exception",
                        url_id=url_id,
                        url=url,
                        attempt=attempt + 1,
                        proxy_mode="proxy" if current_proxy else "direct",
                        error_type=type(e).__name__,
                        error=str(e),
                    )
                    logger.warning(f"fetch_exception url={url} attempt={attempt + 1} err={type(e).__name__}:{e}")
                
                # 如果不是最后一次尝试，等待后重试
                if attempt < self.retry_attempts - 1:
                    delay = self.retry_delay_base * (2 ** attempt)  # 指数退避
                    await asyncio.sleep(delay)
                    self.stats["retries"] += 1
            
            # 所有重试都失败
            self._trace(
                "final_fetch_failed",
                url_id=url_id,
                url=url,
                retries=self.retry_attempts,
                last_error=last_error,
            )
            self.url_pool_builder.update_status(url_id, "failed", last_error)
            self.stats["failed"] += 1
            return None
    
    async def _crawl_batch(self, url_batch: List[Dict]) -> List[Record]:
        """批量爬取 URL"""
        tasks = [self._crawl_url(url_info) for url_info in url_batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        records = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"爬取任务异常: {result}")
            elif result is not None:
                records.append(result)
        
        return records
    
    def _save_records(self, records: List[Record], date_str: Optional[str] = None):
        """保存记录到文件"""
        if not records:
            return
        
        # 如果没有指定日期，使用当前日期
        if not date_str:
            date_str = datetime.now().strftime("%Y%m%d")
        
        # 保存为 JSONL 格式
        output_file = self.output_dir / f"{date_str}_records.jsonl"
        
        with open(output_file, "a", encoding="utf-8") as f:
            for record in records:
                f.write(record.to_json() + "\n")
        
        logger.info(f"已保存 {len(records)} 条记录到 {output_file}")
    
    async def crawl(self, limit: Optional[int] = None):
        """开始爬取"""
        # 获取待爬取的 URL
        urls = self.url_pool_builder.get_pending_urls(limit=limit)
        
        if not urls:
            logger.warning("没有待爬取的 URL")
            return []
        
        # 测试模式限制（仅在明确启用且没有指定limit时）
        # 注意：如果 limit 已指定，则使用 limit，不再应用 test_url_limit
        if self.config.test_mode and limit is None:
            original_count = len(urls)
            urls = urls[:self.config.test_url_limit]
            if len(urls) < original_count:
                logger.info(f"测试模式：只处理前 {len(urls)} 个 URL（共 {original_count} 个待处理）")
        
        total = len(urls)
        self.stats["total"] = total
        
        logger.info(f"开始爬取 {total} 个 URL")
        
        # 分批处理
        batch_size = self.concurrent_crawls * 2  # 每批处理更多，提高效率
        all_records = []
        
        for i in range(0, total, batch_size):
            batch = urls[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total + batch_size - 1) // batch_size
            
            logger.info(f"处理批次 {batch_num}/{total_batches}，共 {len(batch)} 个 URL")
            
            # 控制全局并发
            semaphore = asyncio.Semaphore(self.concurrent_crawls)
            
            async def crawl_with_semaphore(url_info):
                async with semaphore:
                    return await self._crawl_url(url_info)
            
            tasks = [crawl_with_semaphore(url_info) for url_info in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 收集成功的记录
            batch_records = []
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"爬取任务异常: {result}")
                elif result is not None:
                    batch_records.append(result)
            
            # 保存批次记录
            if batch_records:
                self._save_records(batch_records)
                all_records.extend(batch_records)
            
            # 批次间稍作休息
            if i + batch_size < total:
                await asyncio.sleep(1)
        
        logger.info(f"爬取完成: 总计 {total}，成功 {self.stats['success']}，失败 {self.stats['failed']}，重试 {self.stats['retries']} 次")
        
        return all_records
