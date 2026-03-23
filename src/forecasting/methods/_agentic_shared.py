"""Shared helpers for forecasting method implementations."""

from __future__ import annotations

import json
import math
import re
import time
from collections import Counter
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from agent import Agent
from agent.tools import OpenBBTool, SearchTool
from forecasting.contracts import ForecastResult, QuestionRecord
from forecasting.llm import LLMUsage, OpenAIChatModel
from forecasting.memory import FlexExperience
from forecasting.prompts import (
    build_agent_system_prompt,
    build_agent_user_prompt,
    build_direct_user_prompt,
    build_forced_finalizer_messages,
    build_rag_user_prompt,
    forecast_system_prompt,
    format_docs_for_prompt,
)

EPS = 1e-6
DEFAULT_SEARCH_TOP_K = 3
DEFAULT_SEARCH_CONTENT_CHARS = 512
DEFAULT_AGENT_SEARCH_MAX_CALLS = 2


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
    if hasattr(value, "__dict__") and not isinstance(value, type):
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


def run_direct_io_forecast(
    question: QuestionRecord,
    llm: OpenAIChatModel,
    *,
    method_name: str,
    max_tokens: int = 300,
    temperature: float = 0.0,
) -> ForecastResult:
    started = time.perf_counter()
    messages = [
        {"role": "system", "content": forecast_system_prompt()},
        {"role": "user", "content": build_direct_user_prompt(question)},
    ]
    payload, raw_text, usage = llm.chat_json(
        messages, max_tokens=max_tokens, temperature=temperature
    )
    elapsed = time.perf_counter() - started
    parsed = _normalize_final_payload(payload, raw_text)
    return build_result(
        question=question,
        method_name=method_name,
        predicted_prob=parsed["predicted_prob"],
        reasoning_summary=parsed["reasoning_summary"],
        trajectory=[{"step": "final", "raw_response": raw_text}],
        usage=usage,
        latency_sec=elapsed,
    )


def run_naive_rag_forecast(
    question: QuestionRecord,
    llm: OpenAIChatModel,
    search_engine: Any,
    *,
    method_name: str,
    search_top_k: int,
    search_content_chars: int,
    rag_max_per_source_type: int,
    max_tokens: int = 350,
    temperature: float = 0.0,
) -> ForecastResult:
    started = time.perf_counter()
    query = build_rag_query(question)
    search_result = search_engine.search(
        query, time=question["open_time"], limit=search_top_k
    )
    hits = diversify_hits(
        search_result.get("hits", []),
        top_k=search_top_k,
        max_per_source_type=rag_max_per_source_type,
    )
    context = format_docs_for_prompt(hits, content_chars=search_content_chars)
    messages = [
        {"role": "system", "content": forecast_system_prompt()},
        {"role": "user", "content": build_rag_user_prompt(question, query, context)},
    ]
    payload, raw_text, usage = llm.chat_json(
        messages, max_tokens=max_tokens, temperature=temperature
    )
    elapsed = time.perf_counter() - started
    parsed = _normalize_final_payload(payload, raw_text)
    return build_result(
        question=question,
        method_name=method_name,
        predicted_prob=parsed["predicted_prob"],
        reasoning_summary=parsed["reasoning_summary"],
        trajectory=[
            {"step": "retrieve", "query": query, "hits": _trajectory_hits(hits)},
            {"step": "final", "raw_response": raw_text},
        ],
        usage=usage,
        latency_sec=elapsed,
        retrieved_source_types=[_hit_source_type(hit) for hit in hits],
    )


run_bm25_rag_forecast = run_naive_rag_forecast


def run_agentic_forecast(
    question: QuestionRecord,
    llm: OpenAIChatModel,
    search_engine: Any,
    *,
    project_root: Path,
    method_name: str,
    agent_max_steps: int,
    search_top_k: int = DEFAULT_SEARCH_TOP_K,
    injected_memories: list[Any] | None = None,
    flex_memory_tool: Any | None = None,
    flex_preloaded: list[FlexExperience] | None = None,
    code_interpreter_tool: Any | None = None,
) -> ForecastResult:
    started = time.perf_counter()
    code_interpreter_enabled = _question_requires_code_interpreter(question)
    if code_interpreter_enabled and code_interpreter_tool is not None and hasattr(
        code_interpreter_tool, "begin_question"
    ):
        code_interpreter_tool.begin_question(str(question["market_id"]))
    tools: list[Any] = []
    if code_interpreter_enabled:
        tools.append(
            code_interpreter_tool
            or {
                "name": "code_interpreter",
                "work_dir": str(
                    (
                        project_root / ".qwen_agent_workspace" / "forecasting" / method_name
                    ).resolve()
                ),
            }
        )
    tools.extend(
        [
            SearchTool(
                search_client=search_engine,
                limit=search_top_k,
                max_calls=DEFAULT_AGENT_SEARCH_MAX_CALLS,
            ),
            OpenBBTool(),
        ]
    )
    if flex_memory_tool is not None:
        tools.append(flex_memory_tool)
    agent = Agent(
        llm=llm.to_agent_config(),
        tools=tools,
        system_prompt=build_agent_system_prompt(
            question,
            method_name=method_name,
            injected_memories=injected_memories or [],
            flex_preloaded=flex_preloaded or [],
            code_interpreter_enabled=code_interpreter_enabled,
        ),
        max_steps=agent_max_steps,
        raise_on_tool_error=False,
    )
    responses = agent.run_messages(
        build_agent_user_prompt(question, injected_memories=injected_memories or []),
        cuttime=question["open_time"],
    )
    extracted = _extract_agent_outputs(responses, tool_events=agent.get_last_tool_events())
    final_text = Agent.extract_final_content(responses)
    parsed = _normalize_final_payload(_try_parse_json_dict(final_text), final_text)
    usage = LLMUsage(**agent.get_last_usage())
    if _needs_forced_final_answer(final_text):
        forced_payload, forced_text, forced_usage = _force_final_answer(
            question,
            llm,
            trajectory=extracted["trajectory"],
        )
        if forced_text:
            extracted["trajectory"].append({"step": "forced_finalize", "raw_response": forced_text})
        if forced_payload or forced_text:
            parsed = _normalize_final_payload(forced_payload, forced_text)
        usage = usage.plus(forced_usage)
    elapsed = time.perf_counter() - started
    return build_result(
        question=question,
        method_name=method_name,
        predicted_prob=parsed["predicted_prob"],
        reasoning_summary=parsed["reasoning_summary"],
        trajectory=extracted["trajectory"],
        usage=usage,
        latency_sec=elapsed,
        retrieved_source_types=extracted["retrieved_source_types"],
        steps_count=agent.get_last_llm_call_count(),
        tool_usage_counts=extracted["tool_usage_counts"],
    )


def build_rag_query(question: QuestionRecord) -> str:
    description = question["description"].split("\n", 1)[0].strip()
    parts = [question["question"], description, question["domain"]]
    return " ".join(part for part in parts if part)[:400]


def diversify_hits(
    hits: list[dict[str, Any]], *, top_k: int, max_per_source_type: int = 2
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_positions: set[int] = set()
    per_source_type: dict[str, int] = {}
    for position, hit in enumerate(hits):
        source_type = _hit_source_type(hit)
        if per_source_type.get(source_type, 0) >= max_per_source_type:
            continue
        selected.append(hit)
        selected_positions.add(position)
        per_source_type[source_type] = per_source_type.get(source_type, 0) + 1
        if len(selected) >= top_k:
            break
    if len(selected) < top_k:
        for position, hit in enumerate(hits):
            if position in selected_positions:
                continue
            selected.append(hit)
            if len(selected) >= top_k:
                break
    return selected


def build_failed_result(
    question: QuestionRecord, method_name: str, error: str
) -> ForecastResult:
    label = int(question["label"])
    return {
        "market_id": question["market_id"],
        "domain": question["domain"],
        "difficulty": question["difficulty"],
        "open_time": question["open_time"],
        "resolve_time": question["resolve_time"],
        "sample_time": question.get("sample_time"),
        "method_name": method_name,
        "predicted_prob": 0.5,
        "label": label,
        "accuracy": int(label == 1),
        "brier_score": 0.25,
        "log_loss": 0.6931471805599453,
        "trajectory": [{"step": "error", "message": error}],
        "reasoning_summary": "fallback due to runtime error",
        "latency_sec": 0.0,
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "error": error,
    }


def extract_entities_from_question(question: QuestionRecord) -> list[str]:
    import re

    text = f"{question['question']} {question['description']}"
    items = []
    for match in re.findall(r"\b[A-Z][A-Za-z0-9.&'-]{2,}\b|\b[A-Z]{2,6}\b", text):
        cleaned = match.strip(".,:;()[]{}\"'")
        if cleaned and cleaned not in items:
            items.append(cleaned)
        if len(items) >= 8:
            break
    return items


def build_memory_query(question: QuestionRecord) -> str:
    description = " ".join(question["description"].split())[:300]
    return f"{question['question']} {description} {question['domain']}".strip()


def build_result(
    *,
    question: QuestionRecord,
    method_name: str,
    predicted_prob: float,
    reasoning_summary: str,
    trajectory: list[dict[str, Any]],
    usage: LLMUsage,
    latency_sec: float,
    retrieved_source_types: list[str] | None = None,
    steps_count: int | None = None,
    tool_usage_counts: dict[str, int] | None = None,
) -> ForecastResult:
    label = int(question["label"])
    predicted_prob = min(1.0, max(0.0, float(predicted_prob)))
    result: ForecastResult = {
        "market_id": question["market_id"],
        "domain": question["domain"],
        "difficulty": question["difficulty"],
        "open_time": question["open_time"],
        "resolve_time": question["resolve_time"],
        "sample_time": question.get("sample_time"),
        "method_name": method_name,
        "predicted_prob": predicted_prob,
        "label": label,
        "accuracy": int((predicted_prob >= 0.5) == bool(label)),
        "brier_score": (predicted_prob - label) ** 2,
        "log_loss": _log_loss(predicted_prob, label),
        "trajectory": trajectory,
        "reasoning_summary": reasoning_summary,
        "latency_sec": latency_sec,
        **usage.to_dict(),
    }
    source_types = list(dict.fromkeys(retrieved_source_types or []))
    if source_types:
        result["retrieved_source_types"] = source_types
    if steps_count is not None:
        result["steps_count"] = max(0, int(steps_count))
    tool_counts = dict(sorted((tool_usage_counts or {}).items()))
    if tool_counts:
        result["tool_usage_counts"] = tool_counts
    return result


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


def _needs_forced_final_answer(final_text: str) -> bool:
    payload = _try_parse_json_dict(final_text)
    if _extract_probability_from_payload(payload) is not None:
        return False
    return _extract_probability_from_structured_text(final_text) is None


def _force_final_answer(
    question: QuestionRecord,
    llm: OpenAIChatModel,
    *,
    trajectory: list[dict[str, Any]],
) -> tuple[dict[str, Any], str, LLMUsage]:
    messages = build_forced_finalizer_messages(question, trajectory=trajectory)
    try:
        return llm.chat_json(messages, max_tokens=220, temperature=0.0)
    except Exception:
        return {}, "", LLMUsage()


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


def _extract_agent_outputs(
    messages: list[dict[str, Any]],
    *,
    tool_events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    trajectory: list[dict[str, Any]] = []
    retrieved_source_types: list[str] = []
    tool_usage = Counter()
    tool_call_count = 0
    event_index = 0
    for message in messages:
        role = str(message.get("role") or "")
        function_call = message.get("function_call")
        if role == "assistant" and function_call:
            assistant_content = _compact_text(
                _stringify_trace_content(message.get("content")), 600
            )
            if assistant_content:
                trajectory.append(
                    {
                        "step": f"assistant_before_tool_{tool_call_count + 1}",
                        "content": assistant_content,
                    }
                )
            tool_call_count += 1
            trajectory.append(
                {
                    "step": f"tool_call_{tool_call_count}",
                    "tool_name": function_call.get("name"),
                    "arguments": function_call.get("arguments"),
                }
            )
            continue
        if role == "function":
            tool_name = str(message.get("name") or "unknown")
            tool_usage[tool_name] += 1
            payload = _tool_event_payload(
                tool_name,
                tool_events[event_index] if tool_events and event_index < len(tool_events) else None,
                fallback_content=message.get("content"),
            )
            event_index += 1
            result_entry: dict[str, Any] = {
                "step": f"{tool_name}_result_{tool_usage[tool_name]}",
                "tool_name": tool_name,
            }
            if isinstance(payload, dict):
                error = str(payload.get("error") or "").strip()
                if error:
                    result_entry["error"] = _compact_text(error, 240)
            if tool_name == "search":
                hits = payload.get("hits") if isinstance(payload, dict) else None
                if isinstance(hits, list):
                    result_entry["hits"] = _trajectory_hits(hits)
                    for hit in hits:
                        source_type = _hit_source_type(hit)
                        if source_type:
                            retrieved_source_types.append(source_type)
                warning = str(payload.get("warning") or "").strip() if isinstance(payload, dict) else ""
                if warning:
                    result_entry["warning"] = warning
            elif tool_name == "memory":
                hits = payload.get("hits") if isinstance(payload, dict) else None
                if isinstance(hits, list):
                    result_entry["hits"] = _compact_memory_hits(hits)
            elif tool_name == "openbb":
                result_entry.update(_compact_openbb_payload(payload))
            else:
                result_entry["content"] = _compact_text(
                    _stringify_trace_content(message.get("content")), 600
                )
            trajectory.append(result_entry)
            continue
        if role == "assistant":
            trajectory.append(
                {
                    "step": "assistant",
                    "content": _compact_text(
                        _stringify_trace_content(message.get("content")), 600
                    ),
                }
            )
    return {
        "trajectory": trajectory,
        "retrieved_source_types": list(dict.fromkeys(retrieved_source_types)),
        "tool_usage_counts": dict(sorted(tool_usage.items())),
    }


def _tool_event_payload(
    tool_name: str,
    event: dict[str, Any] | None,
    *,
    fallback_content: Any,
) -> dict[str, Any]:
    if event and str(event.get("tool_name") or "") == tool_name:
        raw_result = event.get("raw_result")
        if isinstance(raw_result, dict):
            return raw_result
    return _parse_tool_payload(fallback_content)


def _parse_tool_payload(content: Any) -> dict[str, Any]:
    if isinstance(content, dict):
        return content
    text = _stringify_trace_content(content).strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {"raw_content": text}
    return payload if isinstance(payload, dict) else {"results": payload}


def _stringify_trace_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
            else:
                text = getattr(item, "text", None) or getattr(item, "content", None)
            if text:
                parts.append(str(text))
        return "\n".join(parts)
    return str(content)


def _compact_text(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _compact_hits(
    hits: list[dict[str, Any]],
    *,
    include_content: bool = False,
) -> list[dict[str, Any]]:
    compacted = []
    for hit in hits:
        item = {
            "doc_id": hit.get("doc_id"),
            "source": hit.get("source"),
            "timestamp": hit.get("timestamp"),
            "title": hit.get("title"),
        }
        if include_content:
            item["content"] = _compact_text(str(hit.get("content") or ""), 240)
        compacted.append(item)
    return compacted


def _trajectory_hits(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    preserved = []
    for hit in hits:
        if isinstance(hit, dict):
            preserved.append(dict(hit))
        else:
            preserved.append({"content": str(hit)})
    return preserved


def _hit_source_type(hit: dict[str, Any]) -> str:
    source_type = str(hit.get("source_type") or "").strip()
    if source_type:
        return source_type
    source = str(hit.get("source") or "").strip().lower()
    if "/" in source:
        return source.split("/", 1)[1]
    return source or "unknown"


def _compact_memory_hits(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compacted = []
    for hit in hits:
        compacted.append(
            {
                "zone": hit.get("zone"),
                "level": hit.get("level"),
                "title": hit.get("title"),
                "summary": _compact_text(str(hit.get("summary") or ""), 180),
                "content": _compact_text(str(hit.get("content") or ""), 220),
            }
        )
    return compacted


def _compact_openbb_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"content": _compact_text(str(payload), 320)}
    results = payload.get("results") if isinstance(payload.get("results"), list) else []
    preview = []
    for row in results[:4]:
        if isinstance(row, dict):
            preview.append({key: row[key] for key in list(row)[:6]})
        else:
            preview.append(row)
    return {
        "function": payload.get("requested_function") or payload.get("function"),
        "result_count": payload.get("result_count"),
        "results_preview": preview,
    }


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


def _question_requires_code_interpreter(question: QuestionRecord) -> bool:
    text = " ".join(
        str(question.get(key) or "")
        for key in ("question", "description", "resolution_criteria")
    ).lower()
    numeric_patterns = [
        r"\bprice\b",
        r"\bclose\b",
        r"\btrade\b",
        r"\bpercent(?:age)?\b",
        r"\bbasis points?\b",
        r"\bbps\b",
        r"\bmarket cap\b",
        r"\brevenue\b",
        r"\beps\b",
        r"\bshare\b",
        r"\bprobability\b",
        r"\bodds\b",
        r"\binterval\b",
        r"\brange\b",
        r"\bratio\b",
        r"\bspread\b",
        r"\bcount\b",
        r"\bnumber of\b",
        r"\bmedals?\b",
        r"\bgoals?\b",
        r"\bwins?\b",
        r"\bpoints?\b",
        r"\bfewer than\b",
        r"\bmore than\b",
        r"\bless than\b",
        r"\bat least\b",
        r"\bat most\b",
        r"\babove\b",
        r"\bbelow\b",
        r"\bunder\b",
        r"\bover\b",
        r"\bbetween\b",
        r"[%$€£¥]",
    ]
    return any(re.search(pattern, text) for pattern in numeric_patterns)


def _log_loss(prob: float, label: int) -> float:
    prob = min(1.0 - EPS, max(EPS, prob))
    if label == 1:
        return -math.log(prob)
    return -math.log(1.0 - prob)
