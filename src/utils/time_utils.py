"""Datetime parsing helpers shared across dataset pipelines."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional


def parse_datetime(value: Any) -> Optional[datetime]:
    """Parse mixed datetime/date strings into UTC datetime."""
    if value is None:
        return None

    if isinstance(value, datetime):
        dt = value
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    text = str(value).strip()
    if not text:
        return None

    if "T" not in text and len(text) == 10:
        text = f"{text}T00:00:00+00:00"

    text = text.replace(" ", "T")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    # Handle timezone strings like +00 / -05.
    if len(text) >= 3 and text[-3] in {"+", "-"} and text[-2:].isdigit():
        text = text + ":00"

    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def to_iso_utc(value: Any) -> Optional[str]:
    dt = parse_datetime(value)
    if dt is None:
        return None
    return dt.isoformat().replace("+00:00", "Z")


def to_day(value: Any) -> Optional[str]:
    dt = parse_datetime(value)
    if dt is None:
        return None
    return dt.date().isoformat()

