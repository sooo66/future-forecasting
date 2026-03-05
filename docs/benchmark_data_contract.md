# Info/KB Pipeline v2（当前实现）

## Scope
仅覆盖 `info` 和 `kb` 文本数据；`question` 使用独立格式。

## Unified Outer Schema
统一记录结构见 `src/core/contracts.py` 的 `TextRecord`：

- `id` (required)
- `kind` (`info` | `kb`)
- `source` (required)
- `timestamp` (required, `YYYY-MM-DD`)
- `url` (nullable)
- `payload` (required object)

说明：
- 现在没有 `timestamp_day` 字段；
- 现在没有 `tags` 字段；
- `payload` 是一个独立 key，不会解包到外层；
- `id` 使用稳定哈希，配合 `--resume` 做去重。

## Source Naming
- News: `news/<domain>`
- Blog/Substack: `blog/substack`
- Reddit: `sociomedia/reddit`
- ArXiv: `paper/arxiv`
- OpenStax: `book/openstax`
- World Bank: `report/world_bank`

## Module Payload（当前）
- `info.news`: `title`, `description`, `content`
- `info.blog.substack`: `title`, `author`, `description`, `content`
- `info.sociomedia.reddit`: `subreddit`, `score`, `num_comments`, `title`, `comments`
- `info.paper.arxiv`: `title`, `authors`, `description`, `content`, `categories`
- `kb.book.openstax`: `title`, `page_title`, `content`
- `kb.report.world_bank`: `title`, `content`

## Modules
可用模块（`--modules`）：

- `info.news`
- `info.blog.substack`
- `info.sociomedia.reddit`
- `info.paper.arxiv`
- `kb.book.openstax`
- `kb.report.world_bank`

`--modules all` 等价于以上全部模块。

## CLI（统一入口）
入口：`src/cli.py`

```bash
python src/cli.py run \
  --snapshot s2026_03_static \
  --from 2026-01-01 \
  --to 2026-01-31 \
  --modules all
```

参数说明：
- `--snapshot`：快照目录名（输出到 `data/benchmark/<snapshot>`）。
- `--from` / `--to`：`info.*` 模块日期范围（必填）。
- `--kb-from` / `--kb-to`：`kb.*` 模块日期范围（可选，不传则继承 `--from/--to`）。
- `--modules`：`all` 或逗号分隔子集。
- `--module-workers`：模块级并行度；`0`=自动（选中模块数），`1`=串行，`N`=最多 N 并行。
- `--resume`：按 `_state` 续跑，跳过已写入 `id`。
- `--snapshot-base-dir`：快照根目录，默认 `data/benchmark`。

## 日期语义（重要）
- `info.*` 严格使用 `--from/--to`。
- `kb.report.world_bank` 使用 `--kb-from/--kb-to`（或继承 info 窗口）。
- `kb.book.openstax` 当前是静态 live 语料模式，模块层会忽略日期窗口（日志会提示 `date window ignored`）。

## 写入与并行
- Runner 会在模块 `yield` 每条记录时立刻 append 到对应 `records.jsonl`（边抓边写）。
- `--modules all` 可以通过 `--module-workers` 开启模块并行。

## Snapshot Layout
输出目录：`data/benchmark/<snapshot_id>/`

- `info/news/records.jsonl`
- `info/blog/substack/records.jsonl`
- `info/sociomedia/reddit/records.jsonl`
- `info/paper/arxiv/records.jsonl`
- `kb/book/openstax/records.jsonl`
- `kb/report/world_bank/records.jsonl`
- `_meta/run_manifest.json`
- `_meta/stats.json`
- `_state/<module>.json`
- `_work/<module>/...`（模块中间产物）

## 常用运行方式
1. 全量（info 用短窗口，kb 用单独窗口）：

```bash
python src/cli.py run \
  --snapshot s2026_03_all \
  --from 2026-02-01 \
  --to 2026-02-28 \
  --kb-from 2025-01-01 \
  --kb-to 2025-12-31 \
  --modules all \
  --module-workers 6
```

2. 仅跑 info：

```bash
python src/cli.py run \
  --snapshot s2026_03_info \
  --from 2026-02-01 \
  --to 2026-02-28 \
  --modules info.news,info.blog.substack,info.sociomedia.reddit,info.paper.arxiv \
  --module-workers 4
```

3. 仅跑 kb（OpenStax + World Bank）：

```bash
python src/cli.py run \
  --snapshot s2026_03_kb \
  --from 2026-02-01 \
  --to 2026-02-28 \
  --kb-from 2024-01-01 \
  --kb-to 2025-12-31 \
  --modules kb.book.openstax,kb.report.world_bank \
  --module-workers 2
```

4. 失败后续跑：

```bash
python src/cli.py run \
  --snapshot s2026_03_all \
  --from 2026-02-01 \
  --to 2026-02-28 \
  --kb-from 2025-01-01 \
  --kb-to 2025-12-31 \
  --modules all \
  --module-workers 6 \
  --resume
```
