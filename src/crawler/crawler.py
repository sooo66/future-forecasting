"""Crawl4AI 爬虫模块"""
import asyncio
import json
import os
import random
import re
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
    """用于 LLMExtractionStrategy 的结构化输出定义"""
    
    title: str = Field(..., description="Headline of the article")
    summary: str = Field(..., description="Concise 1-3 sentence summary")
    content: str = Field(..., description="Full clean article body text without navigation or ads")
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
    
    async def _try_archive_is(self, original_url: str, browser_config, run_config) -> Optional[Dict]:
        """尝试使用 archive.is 获取存档内容以绕过 paywall
        
        archive.is 的工作方式：
        1. 访问 https://archive.is/{url} 会显示存档列表
        2. 最新的存档链接通常在页面中，格式类似 /newest/{timestamp} 或直接是完整URL
        3. 访问存档链接后，页面会重定向或显示存档内容
        """
        try:
            # 构建 archive.is 查询URL
            archive_query_url = f"https://archive.is/{original_url}"
            
            async with AsyncWebCrawler(config=browser_config) as crawler:
                # 先访问查询页面
                if run_config:
                    query_result = await crawler.arun(url=archive_query_url, config=run_config)
                else:
                    query_result = await crawler.arun(url=archive_query_url)
                
                if not query_result.success or not query_result.html:
                    return None
                
                # 解析 archive.is 页面，查找存档链接
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(query_result.html, 'html.parser')
                
                # archive.is 的存档链接通常在以下位置：
                # 1. <a> 标签的 href 属性，包含 /newest/ 或时间戳
                # 2. 或者直接是完整的存档URL
                archive_link = None
                
                # 方法1: 查找包含 /newest/ 的链接
                for link in soup.find_all('a', href=True):
                    href = link.get('href', '')
                    if '/newest/' in href or '/web/' in href:
                        if href.startswith('/'):
                            archive_link = f"https://archive.is{href}"
                        elif href.startswith('http'):
                            archive_link = href
                        break
                
                # 方法2: 如果没找到，查找包含原始URL的完整链接
                if not archive_link:
                    for link in soup.find_all('a', href=True):
                        href = link.get('href', '')
                        if original_url in href and 'archive.is' in href:
                            archive_link = href
                            break
                
                # 方法3: 尝试直接访问存档（archive.is 有时会直接显示）
                # 检查当前页面是否已经是存档内容
                if not archive_link:
                    # 检查页面中是否包含原始URL的内容
                    page_text = soup.get_text()
                    if len(page_text) > 1000:  # 如果页面内容足够长，可能是存档内容
                        archive_link = archive_query_url
                
                if archive_link:
                    # 访问存档页面
                    if run_config:
                        archive_result = await crawler.arun(url=archive_link, config=run_config)
                    else:
                        archive_result = await crawler.arun(url=archive_link)
                    
                    if archive_result.success:
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
                        
                        if extracted["content"] and len(extracted["content"].strip()) > 100:
                            return {
                                "content": extracted["content"],
                                "title": extracted["title"],
                                "summary": extracted["summary"]
                            }
        
        except Exception as e:
            logger.debug(f"尝试 archive.is 失败 {original_url}: {e}")
        
        return None

    async def _run_llm_extraction(self, url: str, base_run_config_kwargs: Dict, browser_config) -> Optional[Dict]:
        """在需要时触发 LLM 提取（重用同一套反爬配置）"""
        if not self.llm_strategy:
            return None
        
        llm_run_kwargs = dict(base_run_config_kwargs)
        llm_run_kwargs["extraction_strategy"] = self.llm_strategy
        run_config_llm = CrawlerRunConfig(**llm_run_kwargs)
        
        try:
            async with AsyncWebCrawler(config=browser_config) as crawler:
                llm_result = await crawler.arun(url=url, config=run_config_llm)
            
            if llm_result.success and getattr(llm_result, "extracted_content", None):
                parsed = self._parse_llm_content(llm_result.extracted_content)
                if parsed:
                    logger.info(f"LLM 提取成功: {url}")
                    return parsed
            else:
                logger.warning(f"LLM 提取失败: {getattr(llm_result, 'error_message', 'unknown')}")
        except Exception as e:
            logger.warning(f"LLM 提取异常，忽略: {e}")
        
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
    
    async def _crawl_url(self, url_info: Dict) -> Optional[Record]:
        """爬取单个 URL"""
        url_id = url_info["id"]
        url = url_info["url"]
        domain = url_info["domain"]
        source_name = url_info.get("source_name", "")
        gkg_date = url_info.get("gkg_date")
        themes = url_info.get("themes")
        
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
                    timeout=self.config.browser_timeout * 1000  # 转换为毫秒
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
            for attempt in range(self.retry_attempts):
                try:
                    async with AsyncWebCrawler(config=browser_config) as crawler:
                        # Crawl4AI 的 arun 方法参数可能因版本而异
                        if run_config:
                            result = await crawler.arun(url=url, config=run_config)
                        else:
                            # 如果没有 run_config，直接传递参数
                            result = await crawler.arun(
                                url=url,
                                user_agent=user_agent if user_agent else None,
                                proxy=proxy if proxy else None
                            )
                        
                        if result.success:
                            # 提取内容
                            html = result.html
                            
                            # 处理 markdown - Crawl4AI 的 markdown 可能是对象或字符串
                            # 根据文档，如果使用了 DefaultMarkdownGenerator，markdown 是一个对象
                            if hasattr(result, 'markdown'):
                                if isinstance(result.markdown, str):
                                    markdown = result.markdown
                                elif hasattr(result.markdown, 'markdown'):
                                    # 使用清理后的 markdown（PruningContentFilter 处理过的）
                                    markdown = result.markdown.markdown
                                elif hasattr(result.markdown, 'raw_markdown'):
                                    # 如果没有清理后的，使用原始 markdown
                                    markdown = result.markdown.raw_markdown
                                else:
                                    # 尝试转换为字符串
                                    markdown = str(result.markdown)
                            else:
                                markdown = None

                            extracted = self.extractor.extract(
                                html=html,
                                markdown=markdown,
                                url=url,
                                source_name=source_name,
                                gkg_date=gkg_date,
                                themes=themes
                            )

                            llm_payload = None
                            if self.llm_strategy and self.llm_mode == "always":
                                llm_payload = self._parse_llm_content(getattr(result, "extracted_content", None))
                            
                            # 检查内容是否为空（可能是paywall）
                            content = extracted["content"]
                            if (not content or len(content.strip()) < 100) and llm_payload is None:
                                # 尝试使用 archive.is 绕过 paywall
                                logger.info(f"检测到内容为空或过短，尝试使用 archive.is 获取: {url}")
                                archive_result = await self._try_archive_is(url, browser_config, run_config)
                                
                                if archive_result and archive_result.get("content"):
                                    extracted["content"] = archive_result["content"]
                                    if not extracted["title"] or extracted["title"] == url:
                                        extracted["title"] = archive_result.get("title", extracted["title"])
                                    if not extracted["summary"]:
                                        extracted["summary"] = archive_result.get("summary", extracted["summary"])
                                    content = extracted["content"]
                                    logger.info(f"成功从 archive.is 获取内容: {url}")

                            # LLM fallback: 在规则提取仍不足时触发
                            word_count = len(re.findall(r"[A-Za-z]+", content or ""))
                            if self.llm_strategy and self.llm_mode == "fallback" and word_count < self.extractor.min_word_threshold:
                                llm_payload = await self._run_llm_extraction(url, base_run_config_kwargs, browser_config)
                            
                            if llm_payload:
                                extracted["content"] = llm_payload.get("content") or extracted["content"]
                                extracted["title"] = llm_payload.get("title") or extracted["title"]
                                extracted["summary"] = llm_payload.get("summary") or extracted["summary"]
                                extracted["published_at"] = llm_payload.get("published_at") or extracted["published_at"]
                                extracted["language"] = llm_payload.get("language") or extracted["language"]
                                tags = llm_payload.get("tags")
                                if isinstance(tags, str):
                                    tags = [t.strip() for t in tags.split(",") if t.strip()]
                                if isinstance(tags, list):
                                    extracted["tags"] = tags
                                content = extracted["content"]
                                word_count = len(re.findall(r"[A-Za-z]+", content or ""))
                            
                            if not content or word_count < self.extractor.min_word_threshold:
                                last_error = f"content_too_short ({word_count} words)"
                                logger.info(f"内容过短，标记为失败: {url} ({word_count} words)")
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
                            
                            return record
                        else:
                            error_msg = f"爬取失败: {result.error_message or '未知错误'}"
                            last_error = error_msg
                            logger.warning(f"URL {url} 爬取失败: {error_msg}")
                            
                except Exception as e:
                    last_error = str(e)
                    logger.error(f"URL {url} 爬取时发生异常: {e}")
                
                # 如果不是最后一次尝试，等待后重试
                if attempt < self.retry_attempts - 1:
                    delay = self.retry_delay_base * (2 ** attempt)  # 指数退避
                    await asyncio.sleep(delay)
                    self.stats["retries"] += 1
            
            # 所有重试都失败
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
            return
        
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
