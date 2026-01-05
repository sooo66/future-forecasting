# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Tooling and common commands

### Python environment (uv)
- This project uses a `pyproject.toml` with `[[tool.uv.index]]` and an `uv.lock`, so the preferred workflow is via [`uv`](https://github.com/astral-sh/uv).
- Install / sync dependencies:
  - Full sync: `uv sync`
  - One-off command with dependency resolution: `uv run python -m src.main --help`

### Running the CLI
The main entrypoint is `src/main.py`, which exposes several subcommands around the end-to-end news data pipeline.

Typical invocations (adjust dates and paths via `config/settings.toml`):
- Show CLI help:
  - `uv run python -m src.main --help`
- Download GDELT GKG CSVs for the configured date range:
  - `uv run python -m src.main --config config/settings.toml download-gkg`
- Parse downloaded GKG CSVs into Parquet files:
  - `uv run python -m src.main --config config/settings.toml parse-gkg`
- Build the URL pool database and JSONL from parsed GKG Parquet files:
  - `uv run python -m src.main --config config/settings.toml build-url-pool`
- Run the full pipeline (download → parse → build URL pool → crawl via URL pool):
  - `uv run python -m src.main --config config/settings.toml full-pipeline --limit 100`
- Crawl from an explicit list of URLs (bypassing URL pool), using the progressive crawler directly:
  - `uv run python -m src.main --config config/settings.toml crawl --source data/url_pool/urls.jsonl --output data/processed/records/crawl_results.jsonl`

### Running tests
Tests are written with `pytest` under `tests/` and import the package as `src.*`.
- Run the full test suite:
  - `uv run pytest`
- Run a single test module:
  - `uv run pytest tests/test_crawler.py`
- Run a single test function (example):
  - `uv run pytest tests/test_crawler.py::test_extract_title`

### Linting / formatting
- There is no dedicated linting or formatting tool configured in `pyproject.toml` (no `ruff`, `flake8`, etc.). If you introduce one, prefer wiring it through `uv run <tool>` so it shares the same environment.

## High-level architecture

### Overview
The codebase implements a multi-stage news data pipeline built on top of GDELT GKG and a URL pool, with an asynchronous crawler backed by Crawl4AI:
1. **GDELT download**: Fetch raw GKG CSVs for a date range.
2. **GDELT parsing**: Convert the CSVs into normalized Parquet files with a subset of fields.
3. **URL pool building**: Extract and filter news article URLs, persisting them to an SQLite-based URL pool and a JSONL snapshot.
4. **Crawling**: Use a crawl strategy that combines Crawl4AI, undetected browser automation, proxy rotation, and custom extraction heuristics to fetch article content.
5. **Utilities**: Centralized configuration, logging setup, and record models.

The main Python package is `src/` with these key subpackages:
- `src.gdelt_downloader` — download and parse GDELT GKG data.
- `src.url_pool_builder` — build and maintain the URL pool.
- `src.crawler` — crawling strategies and HTML/metadata extraction.
- `src.utils` — configuration, logging, and data models.
- `tests/` — pytest-based coverage of each major subsystem.

Data and configuration live under:
- `config/settings.toml` — main runtime configuration (dates, concurrency, paths, LLM and proxy settings, whitelist of media domains).
- `data/raw/gkg` — raw GKG CSVs.
- `data/processed/records` — processed Parquet files and crawl outputs.
- `data/url_pool` — SQLite URL pool (`url_pool.db`) plus a JSONL export (`url_pool.jsonl`).
- `logs/` — rotated Loguru logs plus JSONL trace logs for crawler events.

### Configuration and utilities (`src.utils`)

- **`Config` (`src/utils/config.py`)**
  - Loads `config/settings.toml` (or another TOML file via `--config`).
  - Provides typed properties for all key settings:
    - General controls: `start_date`, `end_date`, `concurrent_downloads`, `concurrent_crawls`, retries, request delays, `test_mode`, `test_url_limit`, `use_proxy`, `use_browser`.
    - Paths: `raw_data_dir`, `processed_data_dir`, `url_pool_db`, `log_dir`.
    - Whitelist: `media_domains` (mapping domain → human-readable name) and `whitelist_domains` (set of allowed domains).
    - Browser: `browser_headless`, `browser_timeout`.
    - LLM extraction knobs (if used in future flows): provider, API key env var, instruction text, chunking parameters, temperature, etc.
  - Any code that needs configuration should depend on this class rather than reading TOML directly.

- **`setup_logger` (`src/utils/logger.py`)**
  - Configures Loguru with three sinks:
    - Colored console output at `INFO`.
    - Daily-rotated application logs at `logs/app_{date}.log` (DEBUG and above).
    - JSONL "trace" logs at `logs/crawl_trace_{date}.jsonl` filtered by `record["extra"].get("trace") is True`.
    - Error-only logs at `logs/error_{date}.log`.
  - All CLI entrypoints in `src/main.py` call this once per run; reuse this pattern in new entrypoints.

- **`Record` (`src/utils/models.py`)**
  - Dataclass representing a normalized news record: `id`, `source`, `url`, `title`, `summary`, `content`, `published_at`, `language`, `tags`.
  - Provides `to_dict` / `to_json` and inverse constructors.
  - If you introduce new storage or output formats, prefer converting to/from this model rather than raw dicts.

### GDELT fetch and parse (`src.gdelt_downloader`)

- **`GDELTDownloader` (`src/gdelt_downloader/downloader.py`)**
  - Uses `Config.raw_data_dir`, `start_date`, `end_date`, and `concurrent_downloads`.
  - Iterates day by day, downloading `YYYYMMDD.gkg.csv.zip` from the GDELT GKG endpoint, extracting to `.gkg.csv`, and deleting the ZIP.
  - Maintains a set of already-downloaded days based on existing `.gkg.csv` files to support resume behavior.
  - Uses `ThreadPoolExecutor` + `tqdm` for concurrent downloads.
  - Returns a summary dict `{total, success, failed}` used by the CLI and logging.

- **`GDELTParser` (`src/gdelt_downloader/parser.py`)**
  - Reads all `*.gkg.csv` files in `Config.raw_data_dir` in chunks via pandas.
  - Selects and normalizes key columns: `DATE`, `THEMES`, `SOURCES`, `SOURCEURLS`, plus derived `gkg_date` and record IDs (`GKGRECORDID` / `gkg_record_id`).
  - Writes per-day Parquet files to `Config.processed_data_dir` and deletes the original CSVs.
  - Returns a summary including total files, successes, failures, and total row count.

This stage is the authoritative source for the URL pool; changes to the schema here may require matching changes downstream in `URLPoolBuilder`.

### URL pool (`src.url_pool_builder`)

- **`URLPoolBuilder` (`src/url_pool_builder/builder.py`)**
  - Consumes `*_gkg.parquet` files from `Config.processed_data_dir`.
  - Extracts individual article URLs from `SOURCEURLS` and source domains from `SOURCES`.
  - Normalizes domains via `tldextract` and filters to the whitelist from `Config.whitelist_domains` / `media_domains`.
  - Uses simple heuristics (`_is_english_url`) to keep only English content (path language hints, default-English domains, basic non-English path detection).
  - Writes results into an SQLite database (`Config.url_pool_db`) with table `url_pool` including status, error, timestamps, themes, GKG metadata.
  - Maintains indexes on `domain`, `status`, and `url`.
  - Exports a full snapshot of the pool to `url_pool.jsonl` after each build.
  - Provides helpers:
    - `get_pending_urls(limit=None)` to fetch pending URLs as dicts.
    - `update_status(id, status, error=None)` to mark crawl outcomes.
    - `get_statistics()` to summarize counts by status.

The URL pool is the central coordination point between GDELT-derived metadata and the crawler; new crawler implementations should rely on this DB rather than re-deriving URLs from GKG.

### Crawling layer (`src.crawler`)

- **`ProgressiveEvasionCrawler` (`src/crawler/progressive_evasion_crawler.py`)**
  - Core Crawl4AI-based crawler designed to be resilient to common anti-bot measures.
  - Key responsibilities:
    - Load URLs either from a JSONL source file (`--source`) or from a passed-in list.
    - Configure Crawl4AI with an undetected Playwright strategy (`UndetectedAdapter`), `BrowserConfig` (stealth, user-agent rotation, headers), and optional proxy rotation via `ProxyManager` and `RoundRobinProxyStrategy`.
    - Control concurrency and memory via `MemoryAdaptiveDispatcher` using `Config.concurrent_crawls`.
    - For each result, normalize `markdown` output (`fit_markdown` preferred) and derive:
      - `main_content` (body text from markdown).
      - `title`, `description`, `pubtime` using HTML-oriented regex/meta extraction helpers.
      - Response metadata: `proxy_used`, `latency`, `status_code`, `success` boolean.
    - Detect blocking (HTTP status 403/429/503 or common block phrases) and mark such results as failed with a retry flag.
    - Append each record to an output JSONL file with fsync for durability.

  - High-level helper methods group behavior:
    - `_build_proxy_strategy`, `_build_run_config`, `_build_browser_config`, `_init_crawler` — Crawl4AI wiring.
    - `_normalize_fit_markdown`, `_meta_map`, `_extract_title_general`, `_extract_description_general`, `_extract_pubtime_general` — extraction logic on top of HTML/markdown.
    - `_is_blocked` and `_append_output` — robustness and IO.

- **`NewsCrawler` (`src/crawler/news_crawler.py`)**
  - Thin adapter that bridges the URL pool to `ProgressiveEvasionCrawler`:
    - Reads pending URLs from `url_pool` in SQLite (respecting an optional `limit`).
    - Invokes `ProgressiveEvasionCrawler` with those URLs and a default output path under `processed_data_dir`.
  - This is the mechanism used by the `full-pipeline` CLI command.

- **`ContentExtractor` (`src/crawler/extractor.py`)**
  - Pure-BeautifulSoup-based extractor used by tests and potentially future non-Crawl4AI paths.
  - Responsibilities:
    - Extract title from `og:title` / `twitter:title` / `<title>`.
    - Extract summary/description from common meta tags.
    - Extract `published_at` from meta tags, `<time>` elements, or JSON-LD.
    - Extract main content from `<article>`, `<main>`, or best-guess content containers, after removing nav/header/footer/script/style sections.
    - Derive language and tags from themes and URL paths based on heuristic mappings.
  - Tests (`tests/test_crawler.py`) encode desired behavior around prioritizing `og:title`, preferring article body over navigation-only markdown, and tag inference.

### CLI orchestration (`src/main.py`)

The CLI is the main operational surface area for this project:
- Each subcommand (`download-gkg`, `parse-gkg`, `build-url-pool`, `crawl`, `full-pipeline`) follows the same pattern:
  1. Load `Config` from `--config` (defaults to `config/settings.toml`).
  2. Call `setup_logger(config)` to initialize logging.
  3. Construct the relevant service classes (`GDELTDownloader`, `GDELTParser`, `URLPoolBuilder`, `NewsCrawler`, `ProgressiveEvasionCrawler`).
  4. Execute the operation and log structured summaries.
- `full-pipeline` chains the stages together and logs summary statistics across all steps.
- Any new high-level workflow should mimic this structure so logging and configuration remain consistent.

### Tests (`tests/`)

- `tests/test_crawler.py` — behavior of `ContentExtractor` for titles, summaries, publish times, content, language, and tag extraction.
- `tests/test_gdelt_downloader.py` — GKG parsing via `GDELTParser._parse_gkg_file` and basic URL extraction behavior.
- `tests/test_url_pool_builder.py` — domain extraction, English URL heuristics, URL validation, SOURCEURLS parsing, and basic URL pool DB initialization/statistics.
- `tests/test_utils.py` — serialization and optional-field handling for the `Record` dataclass.

Use these tests as executable documentation for expected behavior when modifying the corresponding modules.