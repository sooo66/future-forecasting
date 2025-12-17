# 新闻爬虫系统 - Future Event Forecasting Data Source

一个生产级别的 Python 爬虫系统，用于从 GDELT Global Knowledge Graph (GKG) 构建"权威且偏见较小"的新闻 URL 池，并使用 Crawl4AI 爬取结构化新闻数据，为未来事件预测提供信息来源。

## 功能特性

- ✅ **自动下载 GDELT GKG 数据**：支持批量下载指定日期范围的 GKG 数据，支持断点续传和并发下载
- ✅ **智能 URL 池构建**：基于媒体白名单过滤，自动去重，支持英文内容过滤
- ✅ **高性能异步爬取**：基于 Crawl4AI 的异步爬虫，支持大规模并发
- ✅ **反爬虫策略**：随机延迟、User-Agent 轮换、代理支持、重试机制
- ✅ **结构化数据提取**：自动提取标题、摘要、正文、发布时间、语言、标签等信息
- ✅ **数据持久化**：支持 JSONL 和 Parquet 格式输出
- ✅ **完整的测试覆盖**：包含单元测试，确保代码质量

## 项目结构

```
future-foreacsting/
├── config/
│   └── settings.toml          # 配置文件
├── data/
│   ├── raw/
│   │   └── gkg/                # 原始 GKG 数据
│   ├── processed/
│   │   └── records/             # 处理后的新闻记录
│   └── url_pool/
│       └── url_pool.db         # URL 池数据库
├── logs/                        # 日志文件
├── src/
│   ├── gdelt_downloader/       # GDELT 下载和解析模块
│   │   ├── downloader.py
│   │   └── parser.py
│   ├── url_pool_builder/       # URL 池构建模块
│   │   └── builder.py
│   ├── crawler/                # 爬虫模块
│   │   ├── crawler.py
│   │   └── extractor.py
│   ├── utils/                  # 工具模块
│   │   ├── config.py
│   │   ├── logger.py
│   │   └── models.py
│   └── main.py                 # 主入口 CLI
├── tests/                       # 单元测试
│   ├── test_gdelt_downloader.py
│   ├── test_url_pool_builder.py
│   ├── test_crawler.py
│   └── test_utils.py
├── main.py                      # 入口脚本
├── requirements.txt             # 依赖列表
└── README.md                    # 本文档
```

## 核心数据流

```
[GDELT GKG 数据源]
    ↓
[下载 GKG 文件] → [解析 GKG 数据] → [提取 URL]
    ↓
[白名单过滤] → [去重] → [构建 URL 池]
    ↓
[Crawl4AI 爬取] → [内容提取] → [结构化数据]
    ↓
[JSONL/Parquet 输出]
```

## 安装

### 1. 克隆项目

```bash
git clone <repository-url>
cd future-foreacsting
```

### 2. 创建虚拟环境

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 配置

编辑 `config/settings.toml`，设置日期范围和其他参数：

```toml
[general]
start_date = "2025-01-01"
end_date = "2025-01-31"
concurrent_downloads = 3
concurrent_crawls = 5
test_mode = false
```

## 使用方法

### 命令行接口

系统提供了多个子命令，可以单独执行或组合使用：

#### 1. 下载 GKG 数据

```bash
python main.py download-gkg
```

#### 2. 解析 GKG 数据

```bash
python main.py parse-gkg
```

#### 3. 构建 URL 池

```bash
python main.py build-url-pool
```

#### 4. 爬取新闻

```bash
python main.py crawl
# 或限制数量（用于测试）
python main.py crawl --limit 100
```

#### 5. 执行完整流程

```bash
python main.py full-pipeline
# 或限制爬取数量
python main.py full-pipeline --limit 100
```

### 配置说明

主要配置项在 `config/settings.toml` 中：

- **日期范围**：`start_date` 和 `end_date`
- **并发控制**：`concurrent_downloads`（下载并发数）、`concurrent_crawls`（爬取并发数）
- **重试配置**：`retry_attempts`（重试次数）、`retry_delay_base`（基础延迟）
- **请求延迟**：`request_delay_min` 和 `request_delay_max`（随机延迟范围）
- **测试模式**：`test_mode`（是否启用测试模式）、`test_url_limit`（测试 URL 数量限制）
- **代理配置**：`use_proxy`（是否使用代理）、`proxy_sources`（代理源列表）

## 媒体白名单

系统内置了 40+ 个权威新闻媒体的域名白名单，包括：

- **中立媒体**：BBC News, Reuters, The Christian Science Monitor 等
- **主流媒体**：CNN, The New York Times, The Washington Post, Bloomberg 等
- **专业媒体**：Politico, ProPublica, Axios 等
- **其他媒体**：Forbes, TIME, Newsweek 等

完整列表见 `config/settings.toml` 中的 `[whitelist]` 部分。

## 数据格式

爬取的数据以 `Record` 格式存储，包含以下字段：

```python
@dataclass
class Record:
    id: str                   # UUID4
    source: str               # 如 "news/bbc.com"
    url: str                  # 新闻 URL
    title: str                # 标题
    summary: Optional[str]    # 摘要
    content: str              # 正文内容
    published_at: str         # ISO8601 格式发布时间
    language: Optional[str]  # 语言代码（如 "en"）
    tags: Optional[List[str]] # 标签列表
```

输出文件格式：`data/processed/records/YYYYMMDD_records.jsonl`

## 反爬虫策略

系统实现了多种反爬虫策略：

1. **随机延迟**：请求间随机延迟 1-3 秒
2. **User-Agent 轮换**：从多个真实浏览器 UA 中随机选择
3. **每域名并发限制**：限制每个域名的并发请求数
4. **代理支持**：支持使用代理（通过 freeproxy）
5. **重试机制**：失败自动重试，使用指数退避策略
6. **浏览器模式**：可选的无头浏览器模式（用于需要 JS 渲染的页面）

## 测试

运行单元测试：

```bash
pytest tests/
```

测试覆盖：
- GKG 文件解析与 URL 提取
- 域名白名单过滤
- URL 去重和批次划分
- 从 HTML 中抽取核心内容

## 日志

日志文件保存在 `logs/` 目录：

- `app_YYYY-MM-DD.log`：所有级别的日志
- `error_YYYY-MM-DD.log`：错误日志

日志会自动轮转和压缩。

## 注意事项

1. **数据量**：GKG 数据量很大，建议先小范围测试（使用 `test_mode = true`）
2. **网络**：下载和爬取需要稳定的网络连接
3. **存储**：确保有足够的磁盘空间存储原始数据和处理结果
4. **合规性**：请遵守目标网站的 robots.txt 和使用条款
5. **代理**：如果遇到 IP 封禁，可以启用代理功能

## 故障排除

### 1. GKG 文件下载失败

- 检查网络连接
- 确认日期范围内的文件是否存在（某些日期可能没有数据）
- 查看日志文件了解详细错误信息

### 2. 爬取失败率高

- 降低并发数（`concurrent_crawls`）
- 增加请求延迟（`request_delay_min` 和 `request_delay_max`）
- 启用代理功能
- 检查目标网站是否有反爬虫措施

### 3. 内容提取不准确

- 某些网站可能需要启用浏览器模式（`use_browser = true`）
- 可以调整 `extractor.py` 中的提取逻辑

## 开发

### 添加新的媒体源

编辑 `config/settings.toml`，在 `[whitelist.media_domains]` 中添加：

```toml
{ name = "媒体名称", domain = "域名.com" }
```

### 自定义内容提取

修改 `src/crawler/extractor.py` 中的提取逻辑，针对特定网站优化。

## 许可证

[根据项目实际情况填写]

## 贡献

欢迎提交 Issue 和 Pull Request！

## 联系方式

[根据项目实际情况填写]

