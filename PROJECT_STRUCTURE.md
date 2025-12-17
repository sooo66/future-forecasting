# 项目结构说明

## 目录树

```
future-foreacsting/
├── config/
│   └── settings.toml              # 主配置文件
├── data/
│   ├── raw/
│   │   └── gkg/                   # 原始 GKG CSV 文件（下载后会被删除）
│   ├── processed/
│   │   └── records/               # 处理后的新闻记录（JSONL 格式）
│   └── url_pool/
│       └── url_pool.db            # URL 池 SQLite 数据库
├── logs/                          # 日志文件目录
│   ├── app_YYYY-MM-DD.log         # 应用日志
│   └── error_YYYY-MM-DD.log       # 错误日志
├── src/
│   ├── __init__.py
│   ├── gdelt_downloader/          # GDELT 下载和解析模块
│   │   ├── __init__.py
│   │   ├── downloader.py          # GKG 数据下载器
│   │   └── parser.py              # GKG 数据解析器
│   ├── url_pool_builder/          # URL 池构建模块
│   │   ├── __init__.py
│   │   └── builder.py             # URL 池构建器
│   ├── crawler/                   # 爬虫模块
│   │   ├── __init__.py
│   │   ├── crawler.py             # 主爬虫类
│   │   └── extractor.py           # 内容提取器
│   ├── utils/                     # 工具模块
│   │   ├── __init__.py
│   │   ├── config.py              # 配置管理
│   │   ├── logger.py              # 日志配置
│   │   └── models.py              # 数据模型
│   └── main.py                    # CLI 主入口
├── tests/                         # 单元测试
│   ├── __init__.py
│   ├── test_gdelt_downloader.py
│   ├── test_url_pool_builder.py
│   ├── test_crawler.py
│   └── test_utils.py
├── main.py                        # 项目入口（重定向到 src.main）
├── requirements.txt               # Python 依赖
├── README.md                      # 项目文档
└── PROJECT_STRUCTURE.md           # 本文件
```

## 核心模块说明

### 1. GDELT 下载和解析模块 (`src/gdelt_downloader/`)

**功能**：
- 从 GDELT 官方服务器下载指定日期范围的 GKG 数据
- 支持并发下载和断点续传
- 解析 GKG CSV 文件，提取相关字段（DATE, SOURCES, SOURCEURLS, Themes）
- 将解析后的数据保存为 Parquet 格式

**关键类**：
- `GDELTDownloader`: 负责下载 GKG 文件
- `GDELTParser`: 负责解析 GKG 文件

### 2. URL 池构建模块 (`src/url_pool_builder/`)

**功能**：
- 从解析后的 GKG 数据中提取 URL
- 根据媒体白名单过滤 URL
- 过滤英文内容 URL
- URL 去重
- 将 URL 池存储在 SQLite 数据库中

**关键类**：
- `URLPoolBuilder`: URL 池构建和管理

### 3. 爬虫模块 (`src/crawler/`)

**功能**：
- 使用 Crawl4AI 进行异步爬取
- 实现多种反爬虫策略
- 从 HTML 中提取结构化内容
- 支持代理和重试机制

**关键类**：
- `NewsCrawler`: 主爬虫类
- `ContentExtractor`: 内容提取器

### 4. 工具模块 (`src/utils/`)

**功能**：
- 配置管理
- 日志配置
- 数据模型定义

**关键类**：
- `Config`: 配置管理器
- `Record`: 新闻记录数据模型

## 数据流

```
1. 用户指定日期范围
   ↓
2. GDELTDownloader 下载 GKG 文件
   ↓
3. GDELTParser 解析 GKG 文件，提取 URL
   ↓
4. URLPoolBuilder 过滤和去重 URL，构建 URL 池
   ↓
5. NewsCrawler 从 URL 池中爬取新闻
   ↓
6. ContentExtractor 提取结构化内容
   ↓
7. 保存为 JSONL 格式文件
```

## 配置文件结构

`config/settings.toml` 包含以下主要配置：

- `[general]`: 通用配置（日期范围、并发数、重试等）
- `[paths]`: 路径配置
- `[whitelist]`: 媒体域名白名单
- `[browser]`: 浏览器配置

## 数据格式

### URL 池数据库结构

表名：`url_pool`

字段：
- `id`: UUID（主键）
- `url`: URL 字符串（唯一）
- `domain`: 域名
- `source_name`: 媒体名称
- `gkg_date`: GKG 日期
- `gkg_record_id`: GKG 记录ID
- `themes`: 主题（分号分隔）
- `status`: 状态（pending/success/failed）
- `last_error`: 最后错误信息
- `created_at`: 创建时间
- `updated_at`: 更新时间

### 新闻记录格式（Record）

JSON 格式：
```json
{
  "id": "uuid4",
  "source": "news/bbc.com",
  "url": "https://www.bbc.com/news/article",
  "title": "Article Title",
  "summary": "Article summary",
  "content": "Full article content",
  "published_at": "2025-01-01T12:00:00Z",
  "language": "en",
  "tags": ["politics", "news"]
}
```

保存为 JSONL 格式：每行一个 JSON 对象

## 使用流程

1. **配置**：编辑 `config/settings.toml`
2. **下载**：`python main.py download-gkg`
3. **解析**：`python main.py parse-gkg`
4. **构建 URL 池**：`python main.py build-url-pool`
5. **爬取**：`python main.py crawl`
6. **或执行完整流程**：`python main.py full-pipeline`

## 扩展性

### 添加新的媒体源

在 `config/settings.toml` 的 `[whitelist.media_domains]` 中添加：
```toml
{ name = "媒体名称", domain = "域名.com" }
```

### 自定义内容提取

修改 `src/crawler/extractor.py` 中的提取逻辑。

### 调整反爬虫策略

修改 `src/crawler/crawler.py` 中的相关参数和逻辑。

