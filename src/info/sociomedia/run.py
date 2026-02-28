"""Run sociomedia pipeline standalone."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import List, Optional

from loguru import logger

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from utils.importer_base import BaseImporter
from utils.models import Record

from .discover import QuoraDiscoverConfig, QuoraDiscoverer
from .quora_importer import QuoraConfig, QuoraAnswerImporter
from .twitter_importer import TwitterConfig, TwitterImporter, collect_records


if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError:
        raise ImportError("Python < 3.11 requires tomli: pip install tomli")


@dataclass(frozen=True)
class SociomediaConfig:
    start_date: str
    end_date: str
    output_dir: Path
    twitter_limit: int = 300
    quora_limit: int = 200
    twitter_accounts: Optional[List[str]] = None
    twitter_rsshub_instances: Optional[List[str]] = None
    quora_keywords: Optional[List[str]] = None


def load_config(path: str | Path) -> SociomediaConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))

    date_range = raw.get("date_range", {})
    start_date = date_range.get("start_date")
    end_date = date_range.get("end_date")

    limits = raw.get("limits", {})
    twitter_limit = int(limits.get("twitter_per_day", 300))
    quora_limit = int(limits.get("quora_per_day", 200))

    seeds = raw.get("seeds", {})
    twitter_accounts = seeds.get("twitter_accounts")
    quora_keywords = seeds.get("quora_keywords")

    rsshub = raw.get("rsshub", {})
    twitter_rsshub_instances = rsshub.get("instances")

    paths = raw.get("paths", {})
    output_dir = Path(paths.get("output_dir", "data/processed/sociomedia"))

    return SociomediaConfig(
        start_date=start_date,
        end_date=end_date,
        output_dir=output_dir,
        twitter_limit=twitter_limit,
        quora_limit=quora_limit,
        twitter_accounts=twitter_accounts,
        twitter_rsshub_instances=twitter_rsshub_instances,
        quora_keywords=quora_keywords,
    )


def run_sociomedia_from_config(
    config_path: str | Path,
    *,
    date_override: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    source: str = "all",
    limit_override: Optional[int] = None,
) -> List[Record]:
    config = load_config(config_path)

    if start_date or end_date:
        if not start_date or not end_date:
            raise ValueError("start_date and end_date must be provided together")
        normalized_start, normalized_end = _normalize_date_range(start_date, end_date)
        config = SociomediaConfig(
            start_date=normalized_start,
            end_date=normalized_end,
            output_dir=config.output_dir,
            twitter_limit=config.twitter_limit,
            quora_limit=config.quora_limit,
            twitter_accounts=config.twitter_accounts,
            twitter_rsshub_instances=config.twitter_rsshub_instances,
            quora_keywords=config.quora_keywords,
        )
    elif date_override:
        normalized_start, normalized_end = _normalize_date_range(date_override, date_override)
        config = SociomediaConfig(
            start_date=normalized_start,
            end_date=normalized_end,
            output_dir=config.output_dir,
            twitter_limit=config.twitter_limit,
            quora_limit=config.quora_limit,
            twitter_accounts=config.twitter_accounts,
            twitter_rsshub_instances=config.twitter_rsshub_instances,
            quora_keywords=config.quora_keywords,
        )
    else:
        normalized_start, normalized_end = _normalize_date_range(
            config.start_date, config.end_date
        )
        config = SociomediaConfig(
            start_date=normalized_start,
            end_date=normalized_end,
            output_dir=config.output_dir,
            twitter_limit=config.twitter_limit,
            quora_limit=config.quora_limit,
            twitter_accounts=config.twitter_accounts,
            twitter_rsshub_instances=config.twitter_rsshub_instances,
            quora_keywords=config.quora_keywords,
        )

    if limit_override:
        if source in ("twitter", "all"):
            twitter_limit = limit_override
        else:
            twitter_limit = config.twitter_limit
        if source in ("quora", "all"):
            quora_limit = limit_override
        else:
            quora_limit = config.quora_limit
        config = SociomediaConfig(
            start_date=config.start_date,
            end_date=config.end_date,
            output_dir=config.output_dir,
            twitter_limit=twitter_limit,
            quora_limit=quora_limit,
            twitter_accounts=config.twitter_accounts,
            twitter_rsshub_instances=config.twitter_rsshub_instances,
            quora_keywords=config.quora_keywords,
        )

    records: List[Record] = []

    if source in ("twitter", "all"):
        twitter_config = TwitterConfig(
            start_date=config.start_date,
            end_date=config.end_date,
            output_path=str(config.output_dir / "twitter.jsonl"),
            per_day_limit=config.twitter_limit,
            accounts=config.twitter_accounts,
            rsshub_instances=config.twitter_rsshub_instances,
        )
        twitter_importer = TwitterImporter(twitter_config)
        twitter_records = collect_records(twitter_importer)
        _write_records(twitter_records, Path(twitter_config.output_path))
        records.extend(twitter_records)
        logger.info(f"Twitter records: {len(twitter_records)}")

    if source in ("quora", "all"):
        quora_config = QuoraConfig(
            start_date=config.start_date,
            end_date=config.end_date,
            output_path=str(config.output_dir / "quora.jsonl"),
        )
        discoverer = QuoraDiscoverer(
            QuoraDiscoverConfig(per_day_limit=config.quora_limit, keywords=config.quora_keywords)
        )
        try:
            urls = discoverer.discover_urls()
        except Exception as exc:
            logger.warning(f"Quora discovery failed: {exc}")
            urls = []
        importer = QuoraAnswerImporter(quora_config)
        quora_records: List[Record] = []
        for url in urls:
            record = importer.fetch_answer(url)
            if record:
                quora_records.append(record)
            if len(quora_records) >= config.quora_limit:
                break
        _write_records(quora_records, Path(quora_config.output_path))
        records.extend(quora_records)
        logger.info(f"Quora records: {len(quora_records)}")

    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sociomedia pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="config/sociomedia_sources.toml",
        help="Config path (default: config/sociomedia_sources.toml)",
    )
    parser.add_argument("--start-date", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--source",
        type=str,
        choices=["twitter", "quora", "all"],
        default="all",
        help="Select source (twitter/quora/all)",
    )
    parser.add_argument("--limit", type=int, help="Per-day limit override")
    args = parser.parse_args()

    records = run_sociomedia_from_config(
        args.config,
        start_date=args.start_date,
        end_date=args.end_date,
        source=args.source,
        limit_override=args.limit,
    )
    logger.info(f"Sociomedia records: {len(records)}")


if __name__ == "__main__":
    main()


def _write_records(records: List[Record], output_path: Path) -> None:
    if not records:
        logger.warning(f"No records to write for {output_path}")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    BaseImporter._write_jsonl(records, output_path)


def _normalize_date_range(start_date: str, end_date: str) -> tuple[str, str]:
    start = _normalize_date(start_date, is_end=False)
    end = _normalize_date(end_date, is_end=True)
    return start, end


def _normalize_date(value: str, *, is_end: bool) -> str:
    if "T" in value or ":" in value:
        return value
    if is_end:
        return f"{value}T23:59:59+00:00"
    return f"{value}T00:00:00+00:00"
