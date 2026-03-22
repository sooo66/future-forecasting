"""Minimal qwen_agent wrapper for this project."""

from __future__ import annotations

import json
import traceback
from typing import Any, Dict, Iterator, List, Optional

from qwen_agent.agents import Assistant
from qwen_agent.llm.base import BaseChatModel
from qwen_agent.llm.schema import ContentItem
from qwen_agent.log import logger
from qwen_agent.tools.base import BaseTool
from qwen_agent.utils.tokenization_qwen import tokenizer as qwen_tokenizer

DEFAULT_SYSTEM_PROMPT = (
    "你是一个用于 future forecasting 的研究 agent。\n"
    "需要历史资料时优先使用 `search`。`search` 会自动遵守外部注入的截止时间。\n"
    "需要行情、指数、汇率或加密货币价格时使用 `openbb`。`openbb` 也会自动遵守同一个截止时间。\n"
    "需要计算、画图、清洗数据或编写临时代码时使用 `code_interpreter`。"
)


class AgentError(RuntimeError):
    """Raised when the agent cannot complete a run."""


class _LocalAssistant(Assistant):
    def __init__(
        self,
        *args: Any,
        raise_on_tool_error: bool = False,
        tool_event_recorder: Any | None = None,
        llm_usage_recorder: Any | None = None,
        **kwargs: Any,
    ) -> None:
        self._raise_on_tool_error = raise_on_tool_error
        self._tool_event_recorder = tool_event_recorder
        self._llm_usage_recorder = llm_usage_recorder
        super().__init__(*args, **kwargs)

    def _call_tool(self, tool_name: str, tool_args: Any = "{}", **kwargs: Any) -> Any:
        if tool_name not in self.function_map:
            return f"Tool {tool_name} does not exists."
        tool = self.function_map[tool_name]
        try:
            tool_result = tool.call(tool_args, **kwargs)
        except Exception:
            if self._raise_on_tool_error:
                raise
            ex = traceback.format_exc()
            logger.warning("Tool `%s` failed:\n%s", tool_name, ex)
            if self._tool_event_recorder is not None:
                self._tool_event_recorder(tool_name, tool_args, {"error": ex}, {"error": ex})
            return ex

        serialized_tool_result = tool_result
        formatter = getattr(tool, "format_result_for_model", None)
        if callable(formatter):
            try:
                serialized_tool_result = formatter(tool_result)
            except Exception:
                logger.warning("Failed to format tool `%s` result for model context.", tool_name)
                serialized_tool_result = tool_result

        if self._tool_event_recorder is not None:
            self._tool_event_recorder(tool_name, tool_args, tool_result, serialized_tool_result)

        if isinstance(serialized_tool_result, str):
            return serialized_tool_result
        if isinstance(serialized_tool_result, list) and all(
            isinstance(item, ContentItem) for item in serialized_tool_result
        ):
            return serialized_tool_result
        return json.dumps(serialized_tool_result, ensure_ascii=False, indent=2)

    def _call_llm(
        self,
        messages: List[Any],
        functions: Optional[List[Dict[str, Any]]] = None,
        stream: bool = True,
        extra_generate_cfg: Optional[dict] = None,
    ) -> Iterator[List[Any]]:
        # qwen_agent's streaming OpenAI-compatible path is brittle for providers that emit empty choices.
        response = super()._call_llm(
            messages=messages,
            functions=functions,
            stream=False,
            extra_generate_cfg=extra_generate_cfg,
        )
        if isinstance(response, list):
            if self._llm_usage_recorder is not None:
                self._llm_usage_recorder(messages, functions, response)
            return iter([response])
        return response


class Agent:
    """Single-agent runner built on qwen_agent."""

    def __init__(
        self,
        llm: Optional[BaseChatModel | Dict[str, Any]],
        tools: Optional[List[dict[str, Any] | BaseTool]] = None,
        system_prompt: Optional[str] = None,
        max_steps: int = 8,
        raise_on_tool_error: bool = False,
    ) -> None:
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.max_steps = max_steps
        self.raise_on_tool_error = raise_on_tool_error
        self._last_usage = _empty_usage_dict()
        self._last_tool_events: list[dict[str, Any]] = []
        self._last_llm_call_count = 0
        if llm is None:
            raise AgentError("LLM config or instance is required for qwen_agent.")

        self.llm = llm
        self._apply_max_steps()
        self._assistant = _LocalAssistant(
            llm=self.llm,
            system_message=self.system_prompt,
            function_list=list(tools or []),
            raise_on_tool_error=raise_on_tool_error,
            tool_event_recorder=self._record_tool_event,
            llm_usage_recorder=self._record_llm_usage,
        )

    def run(
        self,
        user_input: str,
        messages: Optional[List[Dict[str, Any]]] = None,
        *,
        cuttime: str | None = None,
    ) -> str:
        responses = self.run_messages(user_input, messages=messages, cuttime=cuttime)
        return self.extract_final_content(responses)

    def run_messages(
        self,
        user_input: str,
        messages: Optional[List[Dict[str, Any]]] = None,
        *,
        cuttime: str | None = None,
    ) -> list[dict[str, Any]]:
        self._reset_run_state()
        self._apply_max_steps()
        history = list(messages or [])
        history.append({"role": "user", "content": user_input})
        responses = self._collect_final_responses(
            self._assistant.run(messages=history, **self._runtime_kwargs(cuttime=cuttime))
        )
        if responses is None:
            return []
        return [self._serialize_message(message) for message in responses]

    def _apply_max_steps(self) -> None:
        try:
            from qwen_agent import settings as qwen_settings
            from qwen_agent.agents import fncall_agent as fncall_agent_module

            qwen_settings.MAX_LLM_CALL_PER_RUN = self.max_steps
            fncall_agent_module.MAX_LLM_CALL_PER_RUN = self.max_steps
        except Exception:
            return

    @staticmethod
    def extract_final_content(messages: List[Any]) -> str:
        for message in reversed(messages):
            if isinstance(message, dict):
                role = message.get("role")
                content = message.get("content")
            else:
                role = getattr(message, "role", None)
                content = getattr(message, "content", None)
            if role == "assistant":
                return Agent._stringify_content(content)
        return ""

    @staticmethod
    def _stringify_content(content: Any) -> str:
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
            return "\n".join(parts).strip()
        return str(content)

    @staticmethod
    def _serialize_message(message: Any) -> dict[str, Any]:
        if isinstance(message, dict):
            return dict(message)
        if hasattr(message, "model_dump"):
            return message.model_dump()
        return {
            "role": getattr(message, "role", ""),
            "content": Agent._stringify_content(getattr(message, "content", "")),
            "name": getattr(message, "name", None),
            "function_call": getattr(message, "function_call", None),
        }

    @staticmethod
    def _collect_final_responses(response_stream: Iterator[List[Any]]) -> List[Any] | None:
        final_responses = None
        for responses in response_stream:
            final_responses = responses
        return final_responses

    def get_last_usage(self) -> dict[str, int]:
        return dict(self._last_usage)

    def get_last_tool_events(self) -> list[dict[str, Any]]:
        return [dict(event) for event in self._last_tool_events]

    def get_last_llm_call_count(self) -> int:
        return int(self._last_llm_call_count)

    @staticmethod
    def _runtime_kwargs(*, cuttime: str | None) -> dict[str, Any]:
        runtime_kwargs: dict[str, Any] = {}
        if cuttime:
            runtime_kwargs["cuttime"] = cuttime
        return runtime_kwargs

    def _reset_run_state(self) -> None:
        self._last_usage = _empty_usage_dict()
        self._last_tool_events = []
        self._last_llm_call_count = 0

    def _record_tool_event(
        self,
        tool_name: str,
        tool_args: Any,
        raw_result: Any,
        model_result: Any,
    ) -> None:
        self._last_tool_events.append(
            {
                "tool_name": tool_name,
                "tool_args": tool_args,
                "raw_result": raw_result,
                "model_result": model_result,
            }
        )

    def _record_llm_usage(
        self,
        messages: List[Any],
        functions: Optional[List[Dict[str, Any]]],
        response: List[Any],
    ) -> None:
        self._last_llm_call_count += 1
        prompt_tokens = _estimate_tokens(
            {
                "messages": [_usage_payload_from_message(message) for message in messages],
                "functions": functions or [],
            }
        )
        completion_tokens = _estimate_tokens([_usage_payload_from_message(message) for message in response])
        self._last_usage["prompt_tokens"] += prompt_tokens
        self._last_usage["completion_tokens"] += completion_tokens
        self._last_usage["total_tokens"] += prompt_tokens + completion_tokens


def _usage_payload_from_message(message: Any) -> dict[str, Any]:
    if isinstance(message, dict):
        return dict(message)
    if hasattr(message, "model_dump"):
        return message.model_dump()
    return {
        "role": getattr(message, "role", ""),
        "content": getattr(message, "content", ""),
        "name": getattr(message, "name", None),
        "function_call": getattr(message, "function_call", None),
    }


def _estimate_tokens(payload: Any) -> int:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return len(qwen_tokenizer.tokenize(text))


def _empty_usage_dict() -> dict[str, int]:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
