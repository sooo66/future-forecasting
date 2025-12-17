"""内容提取模块"""
import re
from typing import Optional, Dict, Tuple, List
from datetime import datetime
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
from loguru import logger

class ContentExtractor:
    """从 HTML 中提取新闻内容的提取器"""
    
    def __init__(self, min_word_threshold: int = 80):
        # min_word_threshold 用于评估内容质量，低于该值会尝试其他提取策略
        self.min_word_threshold = min_word_threshold
    
    def extract_title(self, html: str, url: str) -> str:
        """提取标题
        
        优先级：
        1. <meta property="og:title">
        2. <title>
        3. <meta name="title">
        4. schema.org headline
        5. <h1>
        """
        soup = BeautifulSoup(html, 'html.parser')
        
        # 尝试 OpenGraph title
        og_title = soup.find('meta', property='og:title')
        if og_title and og_title.get('content'):
            return og_title['content'].strip()
        
        # 尝试 title 标签
        title_tag = soup.find('title')
        if title_tag:
            title = title_tag.get_text().strip()
            if title:
                return title
        
        # 尝试 meta name="title"
        meta_title = soup.find('meta', attrs={'name': 'title'})
        if meta_title and meta_title.get('content'):
            return meta_title['content'].strip()
        
        # 尝试 schema.org headline
        schema_headline = soup.find(attrs={'itemprop': 'headline'})
        if schema_headline:
            return schema_headline.get_text().strip()
        
        # 尝试第一个 h1
        h1 = soup.find('h1')
        if h1:
            return h1.get_text().strip()
        
        # 如果都找不到，返回 URL 作为后备
        logger.warning(f"无法提取标题，使用 URL: {url}")
        return url
    
    def extract_summary(self, html: str) -> Optional[str]:
        """提取摘要
        
        优先级：
        1. <meta name="description">
        2. <meta property="og:description">
        3. schema.org description
        4. 从正文首段生成
        """
        soup = BeautifulSoup(html, 'html.parser')
        
        # 尝试 meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc and meta_desc.get('content'):
            desc = meta_desc['content'].strip()
            if desc:
                return desc
        
        # 尝试 OpenGraph description
        og_desc = soup.find('meta', property='og:description')
        if og_desc and og_desc.get('content'):
            desc = og_desc['content'].strip()
            if desc:
                return desc
        
        # 尝试 schema.org description
        schema_desc = soup.find(attrs={'itemprop': 'description'})
        if schema_desc:
            desc = schema_desc.get_text().strip()
            if desc:
                return desc
        
        # 从正文首段生成（如果正文提取成功）
        # 这个方法会在 extract_content 之后调用，所以这里先返回 None
        return None
    
    def extract_content(self, html: str, markdown: Optional[str] = None, domain: Optional[str] = None, url: Optional[str] = None) -> str:
        """提取正文内容
        
        优先利用 Crawl4AI 生成的 markdown；多个候选内容会按质量打分后择优返回，避免导航/广告噪声或内容缺失。
        """
        candidates: List[Tuple[str, str]] = []
        
        # 如果没有传入 domain，尝试从 URL 中提取
        if not domain and url:
            domain = self._extract_domain_from_url(url)
        
        if markdown:
            cleaned_markdown = self._clean_markdown_content(markdown, domain)
            if cleaned_markdown:
                candidates.append(("markdown", cleaned_markdown))
        
        html_content = self._extract_from_html(html)
        if html_content:
            candidates.append(("html", html_content))
        
        best_content, chosen_source = self._select_best_content(candidates)
        if chosen_source:
            logger.debug(f"内容提取使用 {chosen_source} 源，字数 {len(best_content.split())}")
        
        return best_content

    def _score_content(self, text: str) -> float:
        """为候选正文打分，兼顾长度、重复度和导航噪声占比"""
        if not text:
            return 0.0
        
        cleaned = text.strip()
        words = re.findall(r"[A-Za-z]+", cleaned)
        word_count = len(words)
        if word_count == 0:
            return 0.0
        
        lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
        nav_keywords = [
            "subscribe", "menu", "login", "sign up", "sign in", "newsletter",
            "advertisement", "cookie", "privacy", "terms", "related stories",
            "most read", "trending", "sponsor", "promo", "share",
        ]
        text_lower = cleaned.lower()
        nav_hits = sum(text_lower.count(k) for k in nav_keywords)
        nav_penalty = nav_hits / max(len(lines), 1)
        
        unique_lines = len(set(lines))
        unique_ratio = unique_lines / max(len(lines), 1)
        avg_line_length = sum(len(line) for line in lines) / max(len(lines), 1)
        long_line_bonus = min(avg_line_length / 80, 1.0)
        
        # 评分偏向：字数越多越好，行重复率越低越好，导航占比越低越好
        base_score = word_count * (0.6 + 0.4 * long_line_bonus)
        diversity_bonus = base_score * (0.3 * unique_ratio)
        penalty = nav_penalty * 50
        
        score = base_score + diversity_bonus - penalty
        if word_count < self.min_word_threshold:
            score *= 0.6  # 字数过少的内容降权
        
        return max(score, 0.0)

    def _select_best_content(self, candidates: List[Tuple[str, str]]) -> Tuple[str, str]:
        """从多个候选内容中选择质量最高的"""
        best_content = ""
        best_source = ""
        best_score = 0.0
        
        for source, content in candidates:
            deduped = self._deduplicate_lines(content)
            score = self._score_content(deduped)
            if score > best_score:
                best_score = score
                best_content = deduped
                best_source = source
        
        return best_content, best_source

    def _deduplicate_lines(self, text: str) -> str:
        """合并重复行，移除明显的噪声行"""
        seen = set()
        cleaned_lines = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped in seen:
                continue
            seen.add(stripped)
            cleaned_lines.append(stripped)
        return "\n".join(cleaned_lines)
    
    def _clean_markdown_for_domain(self, markdown: str, domain: str) -> str:
        """针对特定域名的清理规则"""
        content = markdown
        
        # Bloomberg 特定清理
        if 'bloomberg.com' in domain:
            # 移除 Bloomberg 特定的导航和广告
            content = re.sub(r'### Bloomberg.*?### For Customers', '', content, flags=re.DOTALL)
            content = re.sub(r'Bloomberg.*?Connecting decision makers.*?', '', content, flags=re.DOTALL)
            content = re.sub(r'### For Customers.*?### Support', '', content, flags=re.DOTALL)
            content = re.sub(r'LIVE NOW.*?PlayWatch', '', content, flags=re.DOTALL)
            content = re.sub(r'Confidential tip\?.*?New Window', '', content, flags=re.DOTALL)
            content = re.sub(r'BTV\+.*?Submit a Tip', '', content, flags=re.DOTALL)
            content = re.sub(r'Help©.*?All Rights Reserved', '', content, flags=re.DOTALL)
            content = re.sub(r'## We\'ve updated our terms.*?', '', content, flags=re.DOTALL)
        
        # ZeroHedge 特定清理
        if 'zerohedge.com' in domain:
            content = re.sub(r'!\[zerohedge logo\]\(\)', '', content)
            content = re.sub(r'\* Join Premium.*?More!triangle', '', content, flags=re.DOTALL)
            content = re.sub(r'\* Advertise.*?mobile-logo', '', content, flags=re.DOTALL)
            content = re.sub(r'Zerohedge Debates.*?apply\.', '', content, flags=re.DOTALL)
            content = re.sub(r'Alt-Market.*?Visual Combat Banzai7', '', content, flags=re.DOTALL)
            content = re.sub(r'!\[print-icon\]\(\)', '', content)
            content = re.sub(r'# Want to know more\?.*?ZEROHEDGE DIRECTLY TO YOUR INBOX', '', content, flags=re.DOTALL)
            content = re.sub(r'Receive a daily recap.*?inbox\.', '', content, flags=re.DOTALL)
            content = re.sub(r'### Sign up now.*?inbox\.', '', content, flags=re.DOTALL)
            content = re.sub(r'### Today\'s Top Stories.*?Copyright', '', content, flags=re.DOTALL)
        
        # CNN 特定清理
        if 'cnn.com' in domain:
            content = re.sub(r'### CNN values your feedback.*?Thank You!', '', content, flags=re.DOTALL)
            content = re.sub(r'Your effort and contribution.*?appreciated\.', '', content, flags=re.DOTALL)
            content = re.sub(r'2025 Elections.*?5 Things Quiz', '', content, flags=re.DOTALL)
            content = re.sub(r'Your CNN account.*?account', '', content, flags=re.DOTALL)
            content = re.sub(r'Ad Choices.*?All Rights Reserved', '', content, flags=re.DOTALL)
            content = re.sub(r'CNN Sans.*?News Network', '', content, flags=re.DOTALL)
            content = re.sub(r'## Legal Terms and Privacy.*?', '', content, flags=re.DOTALL)
        
        # Forbes 特定清理
        if 'forbes.com' in domain:
            content = re.sub(r'Americans Want To Recycle.*?Paid Program', '', content, flags=re.DOTALL)
            content = re.sub(r'How Innovation Can.*?Paid Program', '', content, flags=re.DOTALL)
            content = re.sub(r'Celebrating Top Fundraisers.*?Paid Program', '', content, flags=re.DOTALL)
            content = re.sub(r'America\'s 2025.*?Paid Program', '', content, flags=re.DOTALL)
            content = re.sub(r'World\'s Billionaires.*?More\.\.\.', '', content, flags=re.DOTALL)
            content = re.sub(r'Editorial Standards.*?Permissions', '', content, flags=re.DOTALL)
            content = re.sub(r'LOADING VIDEO.*?FORBES\' FEATURED Video', '', content, flags=re.DOTALL)
            content = re.sub(r'© 2025 Forbes.*?All Rights Reserved', '', content, flags=re.DOTALL)
        
        # Washington Examiner 特定清理
        if 'washingtonexaminer.com' in domain:
            content = re.sub(r'Election 2025.*?', '', content, flags=re.DOTALL)
            content = re.sub(r'!\[.*?Logo.*?\]\(.*?\)', '', content, flags=re.IGNORECASE)
            content = re.sub(r'Faith, Freedom.*?Equality, Not Elitism', '', content, flags=re.DOTALL)
            content = re.sub(r'## Recommended Stories.*?', '', content, flags=re.DOTALL)
            content = re.sub(r'### Promoted Stories.*?', '', content, flags=re.DOTALL)
            content = re.sub(r'#### Related Content.*?', '', content, flags=re.DOTALL)
            content = re.sub(r'## COMMENTARY.*?', '', content, flags=re.DOTALL)
            content = re.sub(r'## RESTORING AMERICA.*?', '', content, flags=re.DOTALL)
        
        # NYPost 特定清理
        if 'nypost.com' in domain:
            content = re.sub(r'Post Sports\+.*?Email Newsletters', '', content, flags=re.DOTALL)
            content = re.sub(r'!\[.*?Read the Latest.*?\]\(\)', '', content, flags=re.IGNORECASE)
            content = re.sub(r'!\[.*?logo.*?\]\(\)', '', content, flags=re.IGNORECASE)
            content = re.sub(r'#### trending now.*?\)', '', content, flags=re.DOTALL)
            content = re.sub(r'## Explore More.*?\)', '', content, flags=re.DOTALL)
            content = re.sub(r'### Start your day.*?story\.', '', content, flags=re.DOTALL)
            content = re.sub(r'Read Next.*?in US News', '', content, flags=re.DOTALL)
            content = re.sub(r'#### Trending Now.*?comments', '', content, flags=re.DOTALL)
            content = re.sub(r'## Now on Page Six.*?', '', content, flags=re.DOTALL)
            content = re.sub(r'## Now on.*?Decider', '', content, flags=re.DOTALL)
            content = re.sub(r'##\s+Covers.*?', '', content, flags=re.DOTALL)
            content = re.sub(r'## More Stories.*?', '', content, flags=re.DOTALL)
            content = re.sub(r'Sections & Features.*?', '', content, flags=re.DOTALL)
            content = re.sub(r'© 2025 NYP Holdings.*?', '', content, flags=re.DOTALL)
        
        return content
    
    def _clean_markdown_content(self, markdown: str, domain: Optional[str] = None) -> str:
        """清理 markdown 内容，移除导航、链接、标签等，提取干净的主要内容
        
        注意：如果使用了 Crawl4AI 的 PruningContentFilter，大部分导航内容应该已经被过滤
        这里进行额外的清理以确保内容质量
        """
        if not markdown:
            return ""
        
        content = markdown.strip()
        
        # 先应用域名特定的清理规则
        if domain:
            content = self._clean_markdown_for_domain(content, domain)
        
        # 移除代码块（通常不是正文内容）
        content = re.sub(r'```[\s\S]*?```', '', content)
        
        # 移除 markdown 链接格式 [text](url) - 但保留纯文本
        # 先提取链接文本，然后移除链接格式
        def replace_link(match):
            link_text = match.group(1)
            # 如果链接文本看起来像导航项（短且包含常见导航词），则移除
            nav_keywords = ['skip', 'menu', 'login', 'sign', 'register', 'subscribe', 
                          'follow', 'share', 'tweet', 'facebook', 'linkedin', 'instagram',
                          'navigation', 'nav', 'header', 'footer', 'sidebar', 'aside',
                          'cookie', 'privacy', 'terms', 'contact', 'about', 'home',
                          'bloomberg', 'terminal', 'demo', 'request', 'customer', 'support',
                          'anywhere', 'remote', 'login', 'company', 'products']
            link_lower = link_text.lower()
            if any(keyword in link_lower for keyword in nav_keywords):
                return ''
            # 如果链接文本很短（可能是标签），也移除
            if len(link_text) < 3:
                return ''
            # 如果链接文本全是符号或数字，移除
            if re.match(r'^[^\w\s]+$', link_text) or link_text.isdigit():
                return ''
            # 否则保留文本，移除链接格式
            return link_text
        
        content = re.sub(r'\[([^\]]+)\]\([^\)]+\)', replace_link, content)
        
        # 移除纯 URL（http:// 或 https:// 开头的行或文本）
        content = re.sub(r'https?://[^\s\)]+', '', content)
        
        # 移除图片标记 ![...](url) 或 ![alt]
        content = re.sub(r'!\[[^\]]*\]\([^\)]*\)', '', content)
        content = re.sub(r'!\[[^\]]*\]', '', content)
        
        # 移除标签格式，如 * [Trading] * [Compliance] 或 [Trading] * 等
        content = re.sub(r'\*\s*\[[^\]]+\]\s*\*', '', content)
        content = re.sub(r'\[[^\]]+\]\s*\*', '', content)
        content = re.sub(r'\*\s*\[[^\]]+\]', '', content)
        content = re.sub(r'^\s*\[[^\]]+\]\s*$', '', content, flags=re.MULTILINE)
        
        # 移除行首的符号和空格（如 * item, - item, • item）
        content = re.sub(r'^\s*[•\-\*\+]\s+', '', content, flags=re.MULTILINE)
        
        # 移除只有符号的分隔线（如 "---", "***", "===" 等）
        content = re.sub(r'^[=\-\*_]{3,}\s*$', '', content, flags=re.MULTILINE)
        
        # 移除短行（可能是标签或导航项）
        lines = content.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            # 跳过空行
            if not line:
                continue
            # 跳过只有符号的行
            if re.match(r'^[^\w\s]+$', line):
                continue
            # 跳过看起来像导航的短行（包含常见导航词）
            nav_patterns = [
                r'^(skip|menu|login|sign|register|subscribe|follow|share|tweet|facebook|linkedin|instagram|navigation|nav|header|footer|sidebar|aside|cookie|privacy|terms|contact|about|home)',
                r'^[A-Z\s]{1,30}$',  # 全大写的短行（可能是导航标题）
                r'^(bloomberg|terminal|demo|request|customer|support|anywhere|remote|company|products|logo|advertise|premium|channels|partners)',
                r'^(election|sections|features|company|editions|more from)',  # 网站导航
            ]
            if any(re.match(pattern, line, re.IGNORECASE) for pattern in nav_patterns):
                continue
            
            # 跳过看起来像图片描述的行
            if re.match(r'^!?\[.*\]', line) or 'image' in line.lower() or 'logo' in line.lower():
                continue
            # 跳过只有链接的行（已经处理过链接，但可能还有残留）
            if re.match(r'^https?://', line):
                continue
            # 跳过太短的行（可能是标签，少于 10 个字符）
            if len(line) < 10:
                # 但如果包含常见标点，可能是有效内容
                if not re.search(r'[.!?]', line):
                    continue
            cleaned_lines.append(line)
        
        content = '\n'.join(cleaned_lines)
        
        # 尝试提取主要内容段落
        # 找到最长的连续段落块（通常是正文）
        paragraphs = content.split('\n\n')
        if len(paragraphs) > 3:
            # 过滤掉太短的段落（可能是标签或导航）
            meaningful_paragraphs = []
            for p in paragraphs:
                p_clean = p.strip()
                # 保留长度大于 50 的段落
                if len(p_clean) > 50:
                    # 检查段落是否包含太多链接标记（可能是导航）
                    link_count = len(re.findall(r'\[|\]|\(https?://', p_clean))
                    if link_count < len(p_clean) / 20:  # 链接密度不能太高
                        meaningful_paragraphs.append(p_clean)
            
            if meaningful_paragraphs:
                # 找到最长的段落块（通常是正文）
                # 按长度排序，保留前 80% 的段落
                meaningful_paragraphs.sort(key=len, reverse=True)
                keep_count = max(1, int(len(meaningful_paragraphs) * 0.8))
                content = '\n\n'.join(meaningful_paragraphs[:keep_count])
            elif paragraphs:
                # 如果没有找到长段落，至少保留最长的几个
                paragraphs.sort(key=len, reverse=True)
                content = '\n\n'.join(paragraphs[:min(5, len(paragraphs))])
        
        # 移除多余的换行
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        # 移除行首和行尾的多余空格
        lines = content.split('\n')
        content = '\n'.join(line.strip() for line in lines if line.strip())
        content = self._deduplicate_lines(content)
        
        return content.strip()
    
    def _extract_from_html(self, html: str) -> str:
        """从 HTML 中提取主要内容"""
        soup = BeautifulSoup(html, 'html.parser')
        
        # 移除脚本、样式、导航等
        for element in soup(["script", "style", "nav", "header", "footer", "aside", 
                            "noscript", "iframe", "form", "button", "svg"]):
            element.decompose()
        
        # 移除常见的导航和侧边栏类
        removal_selectors = [
            '.nav', '.navigation', '.navbar', '.menu', '.sidebar', '.aside', '.header', '.footer',
            '.breadcrumb', '.tags', '.related', '.comments', '.social', '.share', '.subscribe',
            '.advertisement', '.ad', '.ads', '.promo', '.sponsor', '.newsletter', '.signup',
            '.cookie', '.consent', '.gdpr', '.modal', '.overlay', '[role="navigation"]',
            '[aria-label*="breadcrumb"]', '[aria-label*="navigation"]'
        ]
        for element in soup.select(', '.join(removal_selectors)):
            element.decompose()
        
        # 尝试找到文章主体
        article_selectors = [
            'article',
            '[role="article"]',
            'main',
            '.article-content',
            '.article-body',
            '.post-content',
            '.entry-content',
            '.story-body',
            '.content-body',
            '.story-text',
            '.article-text',
            '[itemprop="articleBody"]'
        ]
        
        article = None
        for selector in article_selectors:
            article = soup.select_one(selector)
            if article:
                break
        
        if not article:
            # 如果没有找到，尝试找到包含最多文本的 div
            body = soup.find('body')
            if body:
                # 找到所有包含文本的 div
                divs = body.find_all('div', recursive=True)
                if divs:
                    # 选择文本最长的 div（通常是正文）
                    article = max(divs, key=lambda d: len(d.get_text(strip=True)))
        
        if not article:
            article = soup.find('body')
        
        if article:
            # 提取文本，移除短行和链接
            text = article.get_text(separator='\n', strip=True)
            
            # 清理文本
            lines = text.split('\n')
            cleaned_lines = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                # 跳过短行（可能是标签）
                if len(line) < 10:
                    continue
                # 跳过看起来像 URL 的行
                if line.startswith('http://') or line.startswith('https://'):
                    continue
                cleaned_lines.append(line)
            
            text = '\n'.join(cleaned_lines)
            # 清理多余的空白
            text = re.sub(r'\n{3,}', '\n\n', text)
            return self._deduplicate_lines(text.strip())
        
        return ""
    
    def extract_published_at(self, html: str, fallback_date: Optional[str] = None) -> str:
        """提取发布时间
        
        优先级：
        1. <meta property="article:published_time">
        2. <time datetime="...">
        3. schema.org datePublished
        4. 回退到 fallback_date（通常是 GKG 日期）
        """
        soup = BeautifulSoup(html, 'html.parser')
        
        # 尝试 OpenGraph published_time
        og_published = soup.find('meta', property='article:published_time')
        if og_published and og_published.get('content'):
            try:
                dt = datetime.fromisoformat(og_published['content'].replace('Z', '+00:00'))
                return dt.strftime('%Y-%m-%dT%H:%M:%SZ')
            except:
                pass
        
        # 尝试 time 标签
        time_tag = soup.find('time', attrs={'datetime': True})
        if time_tag:
            try:
                dt = datetime.fromisoformat(time_tag['datetime'].replace('Z', '+00:00'))
                return dt.strftime('%Y-%m-%dT%H:%M:%SZ')
            except:
                pass
        
        # 尝试 schema.org datePublished
        schema_date = soup.find(attrs={'itemprop': 'datePublished'})
        if schema_date:
            date_str = schema_date.get('content') or schema_date.get_text()
            if date_str:
                try:
                    dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    return dt.strftime('%Y-%m-%dT%H:%M:%SZ')
                except:
                    pass
        
        # 回退到 fallback_date
        if fallback_date:
            try:
                # 假设 fallback_date 是 YYYY-MM-DD 格式
                dt = datetime.strptime(fallback_date, '%Y-%m-%d')
                return dt.strftime('%Y-%m-%dT%H:%M:%SZ')
            except:
                pass
        
        # 如果都失败，使用当前时间
        logger.warning(f"无法提取发布时间，使用当前时间")
        return datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
    
    def extract_language(self, html: str, url: str) -> str:
        """提取语言
        
        优先级：
        1. HTML lang 属性
        2. URL 中的语言标识
        3. 默认返回 'en'
        """
        soup = BeautifulSoup(html, 'html.parser')
        
        # 尝试 HTML lang 属性
        html_tag = soup.find('html')
        if html_tag and html_tag.get('lang'):
            lang = html_tag['lang'].lower()
            # 提取主要语言代码（如 'en-US' -> 'en'）
            lang_code = lang.split('-')[0]
            if lang_code == 'en':
                return 'en'
        
        # 检查 URL 中的语言标识
        url_lower = url.lower()
        if '/en/' in url_lower or url_lower.endswith('/en'):
            return 'en'
        
        # 默认返回英文（因为我们已经在前面的步骤中过滤了非英文 URL）
        return 'en'
    
    def extract_tags(self, themes: Optional[str], url: str) -> Optional[list]:
        """提取粗粒度分类标签（限定类别，避免生成细碎标签）"""
        categories = {
            "politics": [
                "politics", "election", "government", "policy", "geopolitics", "diplomacy"
            ],
            "business_finance": [
                "economy", "economic", "econ", "finance", "business", "market", "markets",
                "banking", "investment", "stock", "trade"
            ],
            "technology": [
                "technology", "tech", "ai", "cyber", "software", "hardware", "internet", "it"
            ],
            "science": [
                "science", "research", "space", "nasa", "astronomy", "physics", "climate"
            ],
            "health": [
                "health", "medical", "medicine", "covid", "virus", "disease", "vaccine"
            ],
            "sports": [
                "sports", "sport", "football", "soccer", "nba", "nfl", "mlb", "olympic"
            ],
            "entertainment": [
                "entertainment", "culture", "movie", "film", "music", "tv", "showbiz", "celebrity"
            ],
            "world": [
                "world", "international", "global", "foreign"
            ],
        }
        
        selected = set()
        
        def match_keyword_list(text: str):
            text_lower = text.lower()
            for cat, keywords in categories.items():
                for kw in keywords:
                    if kw in text_lower:
                        selected.add(cat)
        
        # 1) 从 GKG themes 里匹配关键词
        if themes:
            theme_list = [t.strip() for t in themes.split(';') if t.strip()]
            for t in theme_list:
                match_keyword_list(t)
        
        # 2) URL 路径匹配
        match_keyword_list(url)
        
        return list(selected) if selected else None
    
    def extract(self, html: str, markdown: Optional[str], url: str, 
                source_name: str, gkg_date: Optional[str] = None,
                themes: Optional[str] = None) -> Dict:
        """提取所有内容字段"""
        # 提取域名用于特定清理规则
        domain = self._extract_domain_from_url(url)
        
        title = self.extract_title(html, url)
        summary = self.extract_summary(html)
        content = self.extract_content(html, markdown, domain, url)
        published_at = self.extract_published_at(html, gkg_date)
        language = self.extract_language(html, url)
        tags = self.extract_tags(themes, url)
        
        # 如果摘要为空，从正文首段生成
        if not summary and content:
            paragraphs = content.split('\n\n')
            if paragraphs:
                summary = paragraphs[0][:500]  # 最多 500 字符
        
        return {
            "title": title,
            "summary": summary,
            "content": content,
            "published_at": published_at,
            "language": language,
            "tags": tags
        }
    
    def _extract_domain_from_url(self, url: str) -> Optional[str]:
        """从 URL 中提取域名"""
        try:
            parsed = urlparse(url)
            return parsed.netloc.lower()
        except:
            return None
