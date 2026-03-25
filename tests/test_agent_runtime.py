from __future__ import annotations

from agent import Agent
from forecasting.memory import FlexExperience, RetrievedMemoryItem
from forecasting.methods._agentic_shared import _extract_agent_outputs, _normalize_final_payload, run_agentic_forecast
from forecasting.llm import LLMUsage
from forecasting.question_tools import FlexMemoryTool


class _FakeAssistant:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def run(self, *, messages, **kwargs):
        self.calls.append({"messages": list(messages), "kwargs": dict(kwargs)})
        yield [{"role": "assistant", "content": "partial"}]
        yield [{"role": "assistant", "content": "final answer"}]


def test_agent_passes_runtime_cuttime_without_exposing_it_to_model():
    agent = Agent(llm=object())
    fake_assistant = _FakeAssistant()
    agent._assistant = fake_assistant

    responses = agent.run_messages("搜索关于 elon musk 的新闻", cuttime="2026-01-20T00:00:00Z")

    assert fake_assistant.calls[0]["kwargs"]["cuttime"] == "2026-01-20T00:00:00Z"
    assert Agent.extract_final_content(responses) == "final answer"
    assert responses[-1]["content"] == "final answer"


class _FakeForecastLLM:
    def to_agent_config(self):
        return {"model": "fake-model"}

    def chat_json(self, messages, *, max_tokens=500, temperature=0.0):
        return (
            {"predicted_prob": 0.72, "reasoning_summary": "forced final answer"},
            '{"predicted_prob": 0.72, "reasoning_summary": "forced final answer"}',
            LLMUsage(prompt_tokens=11, completion_tokens=7, total_tokens=18),
        )


class _FakeRuntimeAgent:
    def __init__(self, *args, **kwargs):
        self._usage = {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}
        self._llm_calls = 3
        self._tool_events = [
            {
                "tool_name": "search",
                "raw_result": {
                    "hits": [
                        {
                            "doc_id": "doc-1",
                            "source": "news/cnn",
                            "timestamp": "2026-01-10",
                            "title": "AAPL dipped",
                            "content": "AAPL fell after weak demand guidance.",
                        }
                    ]
                },
            }
        ]

    def run_messages(self, user_input, messages=None, *, cuttime=None):
        return [
            {
                "role": "assistant",
                "content": "我先查一下近期需求走弱的证据。",
                "function_call": {"name": "search", "arguments": '{"question":"AAPL weak demand"}'},
            },
            {"role": "function", "name": "search", "content": '{"count":1,"hits":["AAPL fell after weak demand guidance."]}'},
            {"role": "assistant", "content": ""},
        ]

    def get_last_usage(self):
        return dict(self._usage)

    def get_last_tool_events(self):
        return list(self._tool_events)

    def get_last_llm_call_count(self):
        return self._llm_calls

    @staticmethod
    def extract_final_content(messages):
        return ""


def test_agentic_forecast_forces_final_answer_and_keeps_usage(monkeypatch, tmp_path):
    monkeypatch.setattr("forecasting.methods._agentic_shared.Agent", _FakeRuntimeAgent)
    question = {
        "market_id": "m-1",
        "question": "Will AAPL close above 200?",
        "description": "Synthetic question",
        "resolution_criteria": "Synthetic resolution",
        "domain": "finance",
        "open_time": "2026-01-20T00:00:00Z",
        "resolve_time": "2026-02-01T00:00:00Z",
        "label": 1,
        "difficulty": "easy",
        "sample_time": "2026-01-20T00:00:00Z",
    }

    result = run_agentic_forecast(
        question,
        _FakeForecastLLM(),
        object(),
        project_root=tmp_path,
        method_name="agentic_nomem",
        agent_max_steps=5,
    )

    assert result["predicted_prob"] == 0.72
    assert result["reasoning_summary"] == "forced final answer"
    assert result["total_tokens"] == 48
    assert result["prompt_tokens"] == 31
    assert result["completion_tokens"] == 17
    assert result["steps_count"] == 3
    assert "retrieved_docs" not in result
    assert result["tool_usage_counts"] == {"search": 1}
    assert result["trajectory"][0]["step"] == "assistant_before_tool_1"
    assert "需求走弱" in result["trajectory"][0]["content"]
    assert result["trajectory"][1]["step"] == "tool_call_1"
    assert result["trajectory"][-1]["step"] == "forced_finalize"


def test_normalize_final_payload_prefers_embedded_json_probability():
    raw_text = (
        'Based on the evidence, 90% downside would be required. ```json '
        '{"predicted_prob": 0.02, "reasoning_summary": "Tail-risk only."} ```'
    )

    parsed = _normalize_final_payload({}, raw_text)

    assert parsed["predicted_prob"] == 0.02
    assert parsed["reasoning_summary"] == "Tail-risk only."


def test_normalize_final_payload_does_not_treat_threshold_number_as_probability():
    raw_text = (
        '{\n'
        '  "predicted_prob": 0.98,\n'
        '  "reasoning_summary": "The 8.4 billion threshold is high and likely exceeded."\n'
        '}'
    )

    parsed = _normalize_final_payload({}, raw_text)

    assert parsed["predicted_prob"] == 0.98
    assert "8.4 billion" in parsed["reasoning_summary"]


def test_agentic_forecast_disables_code_interpreter_for_non_numeric_question(monkeypatch, tmp_path):
    captured = {}

    class _CaptureAgent:
        def __init__(self, *, llm, tools, system_prompt, max_steps, raise_on_tool_error):
            captured["tools"] = tools
            captured["system_prompt"] = system_prompt
            self._usage = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}

        def run_messages(self, user_input, messages=None, *, cuttime=None):
            return [{"role": "assistant", "content": '{"predicted_prob": 0.4, "reasoning_summary": "done"}'}]

        def get_last_usage(self):
            return dict(self._usage)

        def get_last_tool_events(self):
            return []

        def get_last_llm_call_count(self):
            return 1

        @staticmethod
        def extract_final_content(messages):
            return messages[-1]["content"]

    monkeypatch.setattr("forecasting.methods._agentic_shared.Agent", _CaptureAgent)
    question = {
        "market_id": "m-plain",
        "question": "Will Russia capture Siversk by December 22?",
        "description": "Synthetic question without a numeric-computation requirement.",
        "resolution_criteria": "Synthetic resolution",
        "domain": "world",
        "open_time": "2026-01-20T00:00:00Z",
        "resolve_time": "2026-02-01T00:00:00Z",
        "label": 0,
        "difficulty": "easy",
        "sample_time": "2026-01-20T00:00:00Z",
    }

    result = run_agentic_forecast(
        question,
        _FakeForecastLLM(),
        object(),
        project_root=tmp_path,
        method_name="agentic_nomem",
        agent_max_steps=5,
    )

    tool_names = []
    for tool in captured["tools"]:
        if isinstance(tool, dict):
            tool_names.append(tool.get("name"))
        else:
            tool_names.append(getattr(tool, "name", None))
    assert "code_interpreter" not in tool_names
    assert "Do not use code_interpreter for this question." in captured["system_prompt"]
    assert result["predicted_prob"] == 0.4


def test_agentic_forecast_uses_sample_time_cutoff_and_logs_memory_context(monkeypatch, tmp_path):
    captured = {}

    class _CaptureAgent:
        def __init__(self, *, llm, tools, system_prompt, max_steps, raise_on_tool_error):
            captured["system_prompt"] = system_prompt
            self._usage = {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}

        def run_messages(self, user_input, messages=None, *, cuttime=None):
            captured["cuttime"] = cuttime
            return [{"role": "assistant", "content": '{"predicted_prob": 0.61, "reasoning_summary": "done"}'}]

        def get_last_usage(self):
            return dict(self._usage)

        def get_last_tool_events(self):
            return []

        def get_last_llm_call_count(self):
            return 1

        @staticmethod
        def extract_final_content(messages):
            return messages[-1]["content"]

    monkeypatch.setattr("forecasting.methods._agentic_shared.Agent", _CaptureAgent)
    question = {
        "market_id": "m-sample",
        "question": "Will AAPL close above 200?",
        "description": "Synthetic question",
        "resolution_criteria": "Synthetic resolution",
        "domain": "finance",
        "open_time": "2026-01-20T00:00:00Z",
        "sample_time": "2026-01-27T00:00:00Z",
        "resolve_time": "2026-02-01T00:00:00Z",
        "label": 1,
        "difficulty": "easy",
    }
    injected = [
        RetrievedMemoryItem(
            memory_id="rb-1#1",
            record_id="rb-1",
            source_question_id="q-1",
            domain="finance",
            source_resolved_time="2026-01-10T00:00:00Z",
            success_or_failure="success",
            title="Memory title",
            description="Reusable cue",
            content="Use structured evidence first.",
        )
    ]
    preloaded = [
        FlexExperience(
            experience_id="flex-1",
            source_question_id="q-2",
            domain="finance",
            zone="golden",
            level="strategy",
            title="Strategy title",
            summary="Summary text",
            content="Content text",
            created_at="2026-01-12T00:00:00Z",
            source_open_time="2026-01-01T00:00:00Z",
            source_resolved_time="2026-01-12T00:00:00Z",
            outcome=1,
            correctness=True,
        )
    ]

    result = run_agentic_forecast(
        question,
        _FakeForecastLLM(),
        object(),
        project_root=tmp_path,
        method_name="flex",
        agent_max_steps=5,
        injected_memories=injected,
        flex_preloaded=preloaded,
    )

    assert captured["cuttime"] == "2026-01-27T00:00:00Z"
    assert "2026-01-27T00:00:00Z" in captured["system_prompt"]
    assert "Search snippets may be noisy, incomplete, stale, or off-topic" in captured["system_prompt"]
    assert result["cutoff_time"] == "2026-01-27T00:00:00Z"
    assert result["injected_memories"][0]["memory_id"] == "rb-1#1"
    assert result["flex_preloaded"][0]["experience_id"] == "flex-1"
    assert result["flex_preloaded"][0]["domain"] == "finance"


def test_extract_agent_outputs_preserves_tool_error_messages():
    messages = [{"role": "function", "name": "search", "content": ""}]
    tool_events = [{"tool_name": "search", "raw_result": {"error": "connection refused"}}]

    extracted = _extract_agent_outputs(messages, tool_events=tool_events)

    assert extracted["trajectory"][0]["step"] == "search_result_1"
    assert extracted["trajectory"][0]["error"] == "connection refused"


def test_extract_agent_outputs_keeps_full_search_hits_in_trajectory():
    long_content = "A" * 500
    messages = [{"role": "function", "name": "search", "content": ""}]
    tool_events = [
        {
            "tool_name": "search",
            "raw_result": {
                "hits": [
                    {
                        "doc_id": "doc-1",
                        "source": "info/news",
                        "source_type": "news",
                        "timestamp": "2026-01-10",
                        "title": "Long article",
                        "content": long_content,
                        "url": "https://example.com/long",
                    }
                ]
            },
        }
    ]

    extracted = _extract_agent_outputs(messages, tool_events=tool_events)

    hit = extracted["trajectory"][0]["hits"][0]
    assert hit["content"] == long_content
    assert hit["url"] == "https://example.com/long"


def test_extract_agent_outputs_keeps_memory_hit_metadata_in_trajectory():
    messages = [{"role": "function", "name": "memory", "content": ""}]
    tool_events = [
        {
            "tool_name": "memory",
            "raw_result": {
                "hits": [
                    {
                        "experience_id": "flex-1",
                        "source_question_id": "q-1",
                        "domain": "finance",
                        "zone": "golden",
                        "level": "strategy",
                        "title": "Strategy title",
                        "summary": "Summary text",
                        "content": "Content text",
                        "correctness": True,
                    }
                ]
            },
        }
    ]

    extracted = _extract_agent_outputs(messages, tool_events=tool_events)

    hit = extracted["trajectory"][0]["hits"][0]
    assert hit["experience_id"] == "flex-1"
    assert hit["source_question_id"] == "q-1"
    assert hit["domain"] == "finance"


def test_flex_memory_tool_enforces_current_domain():
    captured = {}

    class _CaptureLibrary:
        def retrieve(self, query, **kwargs):
            captured["query"] = query
            captured["kwargs"] = dict(kwargs)
            return [
                FlexExperience(
                    experience_id="flex-1",
                    source_question_id="q-1",
                    domain="finance",
                    zone="golden",
                    level="strategy",
                    title="Strategy title",
                    summary="Summary text",
                    content="Content text",
                    created_at="2026-01-12T00:00:00Z",
                    source_open_time="2026-01-01T00:00:00Z",
                    source_resolved_time="2026-01-12T00:00:00Z",
                    outcome=1,
                    correctness=True,
                )
            ]

    tool = FlexMemoryTool(_CaptureLibrary(), cutoff_time="2026-01-27T00:00:00Z", domain="finance")

    payload = tool.call('{"query":"AAPL setup","zone":"golden","level":"strategy","top_k":3}')

    assert captured["query"] == "AAPL setup"
    assert captured["kwargs"]["domain"] == "finance"
    assert payload["domain"] == "finance"
    assert payload["hits"][0]["domain"] == "finance"
