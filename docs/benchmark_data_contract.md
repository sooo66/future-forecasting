# Info/KB Pipeline v2

## Scope
只覆盖 `info` 和 `kb` 文本数据。`question` 单独维护自己的格式。

## Unified Outer Schema
每条记录统一外壳（见 `src/core/contracts.py` 的 `TextRecord`）：

- `id` (required)
- `kind` (`info` | `kb`)
- `source` (required)
- `timestamp` (nullable, ISO8601 UTC)
- `timestamp_day` (required, `YYYY-MM-DD`)
- `url` (nullable)
- `tags` (nullable array)
- `payload` (required object)

约束：
- `timestamp_day` 必须存在；
- 若 `timestamp` 存在，则其日期必须与 `timestamp_day` 一致；
- `id` 使用稳定哈希生成，保证断点续跑时可去重。

## Source Naming
- News: `news/<domain>`
- Blog/Substack: `blog/substack`
- Reddit: `sociomedia/reddit`
- OpenStax: `book/openstax`
- World Bank: `report/world_bank`
- ArXiv: `paper/arxiv`

## Module Payload
- `info.news`: `title`, `content`, `description`
- `info.blog.substack`: `title`, `author`, `description`, `content`
- `info.sociomedia.reddit`: `subreddit`, `score`, `num_comments`, `title`, `selftext`, `comments`
- `info.paper.arxiv`: `title`, `description`, `content` (+ optional meta)
- `kb.book.openstax`: `title`, `content`
- `kb.report.world_bank`: `title`, `content`

## Unified Entry
统一命令入口：`src/cli.py`

```bash
python src/cli.py run \
  --snapshot s2026_03_static \
  --from 2026-01-01 \
  --to 2026-01-31 \
  --modules all \
  --resume
```

支持：
- `--from` / `--to`：按日期范围抓取；
- `--resume`：按模块状态断点续跑；
- `--modules`：模块子集运行（逗号分隔）。

## Snapshot Layout
运行输出写入：`data/benchmark/<snapshot_id>/`

- `info/news/records.jsonl`
- `info/blog/substack/records.jsonl`
- `info/sociomedia/reddit/records.jsonl`
- `info/paper/arxiv/records.jsonl`
- `kb/book/openstax/records.jsonl`
- `kb/report/world_bank/records.jsonl`
- `_meta/run_manifest.json`
- `_meta/stats.json`
- `_state/<module>.json`
