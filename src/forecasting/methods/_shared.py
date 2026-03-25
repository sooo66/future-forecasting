"""Generic helpers shared across all forecasting methods."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, is_dataclass
from typing import Any

from forecasting.contracts import ForecastResult, QuestionRecord
from forecasting.llm import LLMUsage


def coerce_config(value: Any, config_cls: type[Any]) -> Any:
    if isinstance(value, config_cls):
        return value
    if value is None:
        return config_cls()
    if isinstance(value, dict):
        return config_cls(**value)
    raise TypeError(
        f"Unsupported config value for {config_cls.__name__}: {type(value)!r}"
    )


def serialize_config(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "__dict__"):
        return {
            key: serialize_config(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    if isinstance(value, dict):
        return {key: serialize_config(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize_config(item) for item in value]
    return value


def question_cutoff_time(question: QuestionRecord) -> str:
    return str(question.get("sample_time") or question["open_time"]).strip()


def build_result(
    *,
    question: QuestionRecord,
    predicted_prob: float,
    reasoning_summary: str,
    trajectory: list[dict[str, Any]],
    usage: LLMUsage,
    latency_sec: float,
    steps_count: int | None = None,
    error: str | None = None,
) -> ForecastResult:
    label = int(question["label"])
    predicted_prob = min(1.0, max(0.0, float(predicted_prob)))
    result: ForecastResult = {
        "market_id": question["market_id"],
        "domain": question["domain"],
        "difficulty": question["difficulty"],
        "predicted_prob": predicted_prob,
        "label": label,
        "brier_score": (predicted_prob - label) ** 2,
        "trajectory": trajectory,
        "reasoning_summary": reasoning_summary,
        "latency_sec": latency_sec,
        "total_tokens": usage.total_tokens,
    }
    if steps_count is not None:
        result["steps_count"] = max(0, int(steps_count))
    if error is not None:
        result["error"] = error
    return result


def build_failed_result(
    question: QuestionRecord, error: str
) -> ForecastResult:
    label = int(question["label"])
    return {
        "market_id": question["market_id"],
        "domain": question["domain"],
        "difficulty": question["difficulty"],
        "predicted_prob": 0.5,
        "label": label,
        "brier_score": 0.25,
        "trajectory": [{"step": "error", "message": error}],
        "reasoning_summary": "fallback due to runtime error",
        "latency_sec": 0.0,
        "total_tokens": None,
        "error": error,
    }


def _normalize_final_payload(payload: dict[str, Any], raw_text: str) -> dict[str, Any]:
    normalized_payload = payload if isinstance(payload, dict) else {}
    if not normalized_payload:
        normalized_payload = _try_parse_json_dict(raw_text)
    prob_value = _extract_probability_from_payload(normalized_payload)
    if prob_value is None:
        prob_value = _extract_probability_from_structured_text(raw_text)
    if prob_value is None:
        prob_value = 0.5
    prob_value = min(1.0, max(0.0, prob_value))
    reasoning = str(
        normalized_payload.get("reasoning_summary") or normalized_payload.get("summary") or ""
    ).strip()
    if not reasoning:
        reasoning = raw_text.strip()[:400]
    return {"predicted_prob": prob_value, "reasoning_summary": reasoning}


def _extract_probability_from_payload(payload: dict[str, Any]) -> float | None:
    if not isinstance(payload, dict):
        return None
    for key in ("predicted_prob", "probability"):
        value = _coerce_probability_value(payload.get(key))
        if value is not None:
            return value
    return None


def _extract_probability_from_structured_text(text: str) -> float | None:
    candidates = [text, _extract_fenced_json(text), _extract_first_json_object(text)]
    pattern = re.compile(
        r'["\']?(predicted_prob|probability)["\']?\s*[:=]\s*["\']?([0-9]+(?:\.[0-9]+)?%?)',
        flags=re.IGNORECASE,
    )
    for candidate in candidates:
        candidate = (candidate or "").strip()
        if not candidate:
            continue
        match = pattern.search(candidate)
        if not match:
            continue
        value = _coerce_probability_value(match.group(2))
        if value is not None:
            return value
    return None


def _coerce_probability_value(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        percent = stripped.endswith("%")
        if percent:
            stripped = stripped[:-1].strip()
        try:
            number = float(stripped)
        except ValueError:
            return None
        if percent and 0.0 <= number <= 100.0:
            return number / 100.0
        if 0.0 <= number <= 1.0:
            return number
        if 1.0 < number <= 100.0:
            return number / 100.0
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if 0.0 <= number <= 1.0:
        return number
    if 1.0 < number <= 100.0:
        return number / 100.0
    return None


def _try_parse_json_dict(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {}
    candidates = [text]
    fenced = _extract_fenced_json(text)
    if fenced:
        candidates.append(fenced)
    first_object = _extract_first_json_object(text)
    if first_object:
        candidates.append(first_object)
    seen: set[str] = set()
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _extract_fenced_json(text: str) -> str:
    marker = "```json"
    lowered = text.lower()
    start = lowered.find(marker)
    if start < 0:
        return ""
    remainder = text[start + len(marker) :]
    end = remainder.find("```")
    if end < 0:
        return remainder.strip()
    return remainder[:end].strip()


def _extract_first_json_object(text: str) -> str:
    start = text.find("{")
    if start < 0:
        return ""
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return ""


def _compact_text(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."
