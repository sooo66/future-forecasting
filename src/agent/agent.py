"""Qwen-Agent based single-agent runner with tool support."""

from __future__ import annotations

import asyncio
import json
import traceback
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from qwen_agent.agents import Assistant
from qwen_agent.llm import BaseChatModel
from qwen_agent.llm.schema import ContentItem
from qwen_agent.log import logger

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
try:
    from tools import tool  # noqa: F401
except Exception:
    pass

class AgentError(RuntimeError):
    """Raised when the agent cannot complete a run."""


class _LocalAssistant(Assistant):
    def __init__(self, *args: Any, raise_on_tool_error: bool = False, **kwargs: Any) -> None:
        self._raise_on_tool_error = raise_on_tool_error
        super().__init__(*args, **kwargs)

    def _call_tool(self, tool_name: str, tool_args: Any = "{}", **kwargs: Any) -> Any:
        if tool_name not in self.function_map:
            return f"Tool {tool_name} does not exists."
        tool = self.function_map[tool_name]
        try:
            tool_result = tool.call(tool_args, **kwargs)
        except Exception as ex:
            if self._raise_on_tool_error:
                raise
            exception_type = type(ex).__name__
            exception_message = str(ex)
            traceback_info = "".join(traceback.format_tb(ex.__traceback__))
            error_message = (
                f"An error occurred when calling tool `{tool_name}`:\n"
                f"{exception_type}: {exception_message}\n"
                f"Traceback:\n{traceback_info}"
            )
            logger.warning(error_message)
            return error_message

        if isinstance(tool_result, str):
            return tool_result
        if isinstance(tool_result, list) and all(isinstance(item, ContentItem) for item in tool_result):
            return tool_result
        return json.dumps(tool_result, ensure_ascii=False, indent=4)


class Agent:
    """Single-agent runner built on qwen_agent."""

    def __init__(
        self,
        llm: Optional[BaseChatModel | Dict[str, Any]],
        tools: Optional[List[str | Dict[str, Any] | BaseTool]] = None,
        system_prompt: Optional[str] = None,
        max_steps: int = 8,
        raise_on_tool_error: bool = False,
    ) -> None:
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.raise_on_tool_error = raise_on_tool_error
        if llm is None:
            raise AgentError("LLM config or instance is required for qwen_agent.")
        self.llm = llm
        self._apply_max_steps()
        self._assistant = _LocalAssistant(
            llm=self.llm,
            system_message=self.system_prompt,
            function_list=list(tools or []),
            raise_on_tool_error=raise_on_tool_error,
        )

    def run(self, user_input: str, messages: Optional[List[Dict[str, Any]]] = None) -> str:
        history = list(messages or [])
        history.append({"role": "user", "content": user_input})
        responses = None
        for responses in self._assistant.run(messages=history):
            pass
        if responses is None:
            return ""
        return self._extract_final_content(responses)

    async def run_async(self, user_input: str, messages: Optional[List[Dict[str, Any]]] = None) -> str:
        return await asyncio.to_thread(self.run, user_input, messages)

    def _apply_max_steps(self) -> None:
        try:
            from qwen_agent import settings as qwen_settings

            qwen_settings.MAX_LLM_CALL_PER_RUN = self.max_steps
        except Exception:
            return

    @staticmethod
    def _extract_final_content(messages: List[Any]) -> str:
        for message in reversed(messages):
            if isinstance(message, dict):
                role = message.get("role")
                content = message.get("content")
            else:
                role = getattr(message, "role", None)
                content = getattr(message, "content", None)
            if role == "assistant":
                return content or ""
        return ""


if __name__ == "__main__":
    llm_cfg = {
        "model": "olmo-3:7b-instruct",
        "model_server": 'http://localhost:11434/v1/',
    }

    tools = ["echo"]

    agent = Agent(llm=llm_cfg, max_steps=5, raise_on_tool_error=True)

    while True:
        user_input = input("User: ")
        if user_input.lower() in {"exit", "quit"}:
            break
        response = agent.run(user_input)
        print(f"Agent: {response}")
