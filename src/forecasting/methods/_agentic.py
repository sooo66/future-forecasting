"""Agentic forecasting method shared implementation."""

from __future__ import annotations

import json
import re
import time
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
    build_rag_user_prompt,
    forecast_system_prompt,
    format_docs_for_prompt,
)

from forecasting.methods._shared import (
    _compact_text,
    _normalize_final_payload,
    build_result,
    question_cutoff_time,
)


# Runtime-configurable defaults. Override via method_configs in experiment spec.
# Example: {"agentic_nomem": {"search_top_k": 5, "agent_max_steps": 10}}
DEFAULT_SEARCH_TOP_K = 3
DEFAULT_SEARCH_CONTENT_CHARS = 512
DEFAULT_AGENT_SEARCH_MAX_CALLS = 2


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
    cutoff_time = question_cutoff_time(question)
    search_result = search_engine.search(
        query, time=cutoff_time, limit=search_top_k
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
        predicted_prob=parsed["predicted_prob"],
        reasoning_summary=parsed["reasoning_summary"],
        trajectory=[
            {"step": "retrieve", "query": query, "hits": _trajectory_hits(hits)},
            {"step": "final", "raw_response": raw_text},
        ],
        usage=usage,
        latency_sec=elapsed,
    )


def run_agentic_forecast(
    question: QuestionRecord,
    llm: OpenAIChatModel,
    search_engine: Any,
    *,
    project_root: Path,
    method_name: str,
    agent_max_steps: int,
    max_tokens: int = 2048,
    search_top_k: int = DEFAULT_SEARCH_TOP_K,
    injected_memories: list[Any] | None = None,
    flex_memory_tool: Any | None = None,
    flex_preloaded: list[FlexExperience] | None = None,
) -> ForecastResult:
    started = time.perf_counter()
    cutoff_time = question_cutoff_time(question)
    tools: list[Any] = [
        SearchTool(
            search_client=search_engine,
            limit=search_top_k,
            max_calls=DEFAULT_AGENT_SEARCH_MAX_CALLS,
        ),
        OpenBBTool(),
    ]
    if flex_memory_tool is not None:
        tools.append(flex_memory_tool)
    agent_config = llm.to_agent_config()
    agent_config["generate_cfg"]["max_tokens"] = max_tokens
    agent = Agent(
        llm=agent_config,
        tools=tools,
        system_prompt=build_agent_system_prompt(
            question,
            method_name=method_name,
            injected_memories=injected_memories or [],
            flex_preloaded=flex_preloaded or [],
        ),
        max_steps=agent_max_steps,
        raise_on_tool_error=False,
    )
    responses = agent.run_messages(
        build_agent_user_prompt(question, injected_memories=injected_memories or []),
        cuttime=cutoff_time,
    )
    extracted = _extract_agent_outputs(responses, tool_events=agent.get_last_tool_events())
    final_text = Agent.extract_final_content(responses)
    parsed = _normalize_final_payload(_try_parse_json_dict(final_text), final_text)
    usage = LLMUsage(**agent.get_last_usage())
    elapsed = time.perf_counter() - started
    return build_result(
        question=question,
        predicted_prob=parsed["predicted_prob"],
        reasoning_summary=parsed["reasoning_summary"],
        trajectory=extracted["trajectory"],
        usage=usage,
        latency_sec=elapsed,
        steps_count=agent.get_last_llm_call_count(),
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


def build_memory_query(question: QuestionRecord) -> str:
    description = " ".join(question["description"].split())[:300]
    return f"{question['question']} {description} {question['domain']}".strip()


def _extract_agent_outputs(
    messages: list[dict[str, Any]],
    *,
    tool_events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    trajectory: list[dict[str, Any]] = []
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
            payload = _tool_event_payload(
                tool_name,
                tool_events[event_index] if tool_events and event_index < len(tool_events) else None,
                fallback_content=message.get("content"),
            )
            event_index += 1
            result_entry: dict[str, Any] = {
                "step": f"{tool_name}_result_{event_index}",
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
                warning = str(payload.get("warning") or "").strip() if isinstance(payload, dict) else ""
                if warning:
                    result_entry["warning"] = warning
            elif tool_name == "memory":
                hits = payload.get("hits") if isinstance(payload, dict) else None
                if isinstance(hits, list) and hits:
                    # Distinguish reasoningbank (has memory_id) vs flex (has level)
                    if "memory_id" in hits[0]:
                        # reasoningbank: single mem object
                        result_entry["mem"] = _compact_text(
                            str(hits[0].get("content") or ""), 400
                        )
                    else:
                        # flex: hits with level and content
                        result_entry["hits"] = [
                            {
                                "level": hit.get("level"),
                                "content": _compact_text(str(hit.get("content") or ""), 400),
                            }
                            for hit in hits
                        ]
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
    return {"trajectory": trajectory}


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


def _trajectory_hits(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    preserved = []
    for hit in hits:
        if isinstance(hit, dict):
            preserved.append({
                "content": hit.get("content"),
                "source": hit.get("source"),
            })
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


# --- JSON parsing helpers ---

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
