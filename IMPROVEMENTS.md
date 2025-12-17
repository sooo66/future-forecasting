# 内容提取优化改进说明

## 问题分析

根据 `data/processed/records/20251210_records.jsonl` 的分析，发现了以下问题：

1. **Content 字段包含大量无关内容**：导航链接、logo、标签等
2. **URL 处理不完整**：从 1200 个 URL 只处理了 88 条记录
3. **Paywall 问题**：部分网站（如 Bloomberg, Forbes）内容为空

## 已实施的优化方案

### 1. 使用 Crawl4AI 的 PruningContentFilter

根据 [Crawl4AI 文档](https://docs.crawl4ai.com/)，我们使用了 `PruningContentFilter` 来自动识别并移除非主要内容：

```python
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

content_filter = PruningContentFilter()
markdown_generator = DefaultMarkdownGenerator(content_filter=content_filter)
```

**功能**：
- 自动识别并移除导航栏、侧边栏、页脚等
- 保留主要内容区域
- 过滤掉低质量内容块

### 2. 配置排除选择器

尝试配置排除选择器来移除更多无关元素：

```python
exclude_selectors = [
    'nav', 'header', 'footer', 'aside', '.nav', '.navigation', 
    '.navbar', '.menu', '.sidebar', '.advertisement', '.ad',
    '.social', '.share', '.subscribe', '.newsletter', '.cookie',
    '.breadcrumb', '.tags', '.related', '.comments', '.trending'
]
```

### 3. 针对特定域名的清理规则

为常见新闻网站添加了特定的清理规则：

- **Bloomberg**: 移除导航、广告、版权信息
- **ZeroHedge**: 移除 logo、导航菜单、订阅提示
- **CNN**: 移除反馈表单、账户登录提示
- **Forbes**: 移除付费内容提示、导航菜单
- **NYPost**: 移除社交媒体链接、相关文章推荐
- **Washington Examiner**: 移除导航、推荐故事等

### 4. Archive.is 支持绕过 Paywall

实现了自动检测空内容并使用 archive.is 获取存档的功能：

- 当检测到内容为空或过短（< 100 字符）时，自动尝试 archive.is
- 支持多种 archive.is 链接查找方式
- 如果存档成功，使用存档内容替换原始内容

### 5. 增强的 Markdown 清理逻辑

改进了通用的 markdown 清理逻辑：

- **移除图片标记**：`![...](url)` 和 `![alt]`
- **移除导航链接**：识别并移除包含导航关键词的链接
- **移除标签格式**：如 `* [Trading] * [Compliance]`
- **移除短行和导航项**：过滤掉太短的行和包含导航关键词的行
- **智能段落提取**：保留最长的段落块作为正文
- **链接密度过滤**：移除链接密度过高的段落（可能是导航栏）

### 6. 修复 URL 处理逻辑

- 改进了测试模式的日志输出
- 确保即使 content 为空也保存记录
- 修复了 markdown 对象处理逻辑（支持 Crawl4AI 的 markdown 对象）

### 7. 可选的 LLM 提取（Crawl4AI LLMExtractionStrategy）

- 配置项 `[llm]` 支持 always/fallback 两种模式；默认关闭，fallback 模式仅在正文过短时触发
- 使用 `LLMExtractionStrategy` + Pydantic schema 让模型输出标准化 JSON（title/summary/content/published_at/language/tags）
- 指令、分块参数、输入格式（markdown/fit_markdown/html）和温度/max_tokens 可配置
- 需要在环境变量中提供 API Key（默认 `OPENAI_API_KEY`）；若未设置自动回退到规则提取

## 使用建议

### 1. 关闭测试模式

在 `config/settings.toml` 中设置：

```toml
test_mode = false
```

这样可以处理所有待处理的 URL（目前有 974 个 pending）。

### 2. 重新运行爬取

```bash
python main.py crawl
```

### 3. 检查结果

查看 `data/processed/records/` 目录下的 JSONL 文件，确认：
- Content 字段是否更干净
- 是否成功绕过了 paywall
- 所有 URL 是否都被处理

## 技术细节

### Crawl4AI Markdown 对象处理

Crawl4AI 的 `result.markdown` 可能是一个对象，包含多个属性：
- `result.markdown.markdown` - 清理后的 markdown（使用 PruningContentFilter）
- `result.markdown.raw_markdown` - 原始 markdown
- `result.markdown.markdown_with_citations` - 带引用的 markdown

代码已支持所有这些格式。

### Archive.is 实现

Archive.is 的工作流程：
1. 访问 `https://archive.is/{original_url}` 查询存档
2. 查找存档链接（通常包含 `/newest/` 或时间戳）
3. 访问存档页面并提取内容
4. 如果成功，使用存档内容替换原始内容

## 预期效果

实施这些优化后，应该能够：

1. ✅ **显著减少无关内容**：PruningContentFilter + 特定清理规则
2. ✅ **绕过部分 Paywall**：自动使用 archive.is
3. ✅ **处理所有 URL**：关闭测试模式后处理所有待处理 URL
4. ✅ **提高内容质量**：更干净的正文内容，适合后续 NLP 处理

## 注意事项

1. **Archive.is 限制**：Archive.is 可能没有所有 URL 的存档，某些 URL 可能仍然无法获取内容
2. **网站结构变化**：如果网站结构发生变化，可能需要更新特定清理规则
3. **性能影响**：使用 archive.is 会增加爬取时间，因为需要额外的请求
4. **内容质量**：即使使用了所有优化，某些网站的内容质量可能仍然不理想

## 后续优化建议

1. **监控内容质量**：定期检查提取的内容，识别新的噪声模式
2. **扩展特定规则**：为更多网站添加特定清理规则
3. **使用 LLM 提取策略**：对于复杂页面，可以考虑使用 Crawl4AI 的 LLM 提取策略
4. **内容验证**：添加内容质量评分，过滤掉质量过低的内容
