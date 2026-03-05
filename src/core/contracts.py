"""Core data contracts for benchmark text records."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Optional

from utils.time_utils import to_day


VALID_KINDS = {"info", "kb"}


@dataclass
class TextRecord:
    """Unified outer schema for info/kb text records."""

    id: str
    kind: str  # info | kb
    source: str
    timestamp: str  # YYYY-MM-DD
    url: Optional[str]
    payload: dict[str, Any]

    def normalized(self) -> "TextRecord":
        day = to_day(self.timestamp) or self.timestamp
        rec = TextRecord(
            id=self.id,
            kind=self.kind,
            source=self.source,
            timestamp=day or "",
            url=self.url,
            payload=self.payload,
        )
        rec.validate()
        return rec

    def validate(self) -> None:
        if not self.id:
            raise ValueError("record.id is required")
        if self.kind not in VALID_KINDS:
            raise ValueError(f"record.kind must be one of {sorted(VALID_KINDS)}")
        if not self.source:
            raise ValueError("record.source is required")
        if not self.timestamp:
            raise ValueError("record.timestamp is required")
        parsed_day = to_day(self.timestamp)
        if parsed_day is None or parsed_day != self.timestamp:
            raise ValueError("record.timestamp must be YYYY-MM-DD")
        if not isinstance(self.payload, dict):
            raise ValueError("record.payload must be object")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


def stable_record_id(*parts: Any) -> str:
    joined = "|".join(str(x or "") for x in parts)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()
