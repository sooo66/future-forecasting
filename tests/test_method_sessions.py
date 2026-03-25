from __future__ import annotations

from forecasting.contracts import MethodRuntimeContext
from forecasting.llm import LLMUsage
from forecasting.methods.flex import FlexMethod
from forecasting.methods.reasoningbank import ReasoningBankMethod


class _FakeLLM:
    model = "fake-model"
    base_url = "https://fake.local"

    def __init__(self) -> None:
        self.chat_calls = 0
        self.chat_json_calls = 0

    def chat(self, messages, *, max_tokens=500, temperature=0.0):
        self.chat_calls += 1
        return (
            "# Memory Item 1\n"
            "## Title Reusable cue\n"
            "## Description One sentence summary.\n"
            "## Content Preserve the useful reasoning pattern.\n",
            LLMUsage(),
        )

    def chat_json(self, messages, *, max_tokens=500, temperature=0.0):
        self.chat_json_calls += 1
        return (
            {
                "strategy": {
                    "title": "Strategy",
                    "summary": "High-level guidance",
                    "content": "Use structured evidence first.",
                },
                "pattern": {
                    "title": "Pattern",
                    "summary": "Reusable pattern",
                    "content": "Check core evidence before updating.",
                },
                "case": {
                    "title": "Case",
                    "summary": "Concrete case note",
                    "content": "This case used structured evidence well.",
                },
            },
            "{}",
            LLMUsage(),
        )


class _FakeCodeInterpreter:
    def __init__(self, work_dir):
        self.work_dir = work_dir

    def begin_question(self, question_key):
        return None


class _FakeEmbedder:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        outputs: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            values = [
                1.0 if "m-1" in lowered else 0.0,
                1.0 if "m-2" in lowered else 0.0,
                1.0 if "m-3" in lowered else 0.0,
            ]
            norm = sum(value * value for value in values) ** 0.5 or 1.0
            outputs.append([value / norm for value in values])
        return outputs


def _runtime_ctx(tmp_path, llm=None):
    output_dir = tmp_path / "artifacts" / "test"
    output_dir.mkdir(parents=True, exist_ok=True)
    llm_instance = llm or _FakeLLM()
    return MethodRuntimeContext(
        project_root=tmp_path,
        output_dir=output_dir,
        search_engine=object(),
        llm_factory=lambda: llm_instance,
    )


def _question(
    market_id: str,
    *,
    open_time: str,
    resolve_time: str,
    label: int,
    sample_time: str | None = None,
    domain: str = "finance",
) -> dict[str, object]:
    return {
        "market_id": market_id,
        "question": f"Question {market_id}",
        "description": "Synthetic question",
        "resolution_criteria": "Synthetic resolution",
        "domain": domain,
        "open_time": open_time,
        "resolve_time": resolve_time,
        "resolved_time": resolve_time,
        "sample_time": sample_time or open_time,
        "difficulty": "easy",
        "sampled_prob_yes": 0.5,
        "label": label,
        "horizon": "7d",
    }


def test_reasoningbank_session_does_not_leak_future_or_cross_domain_memory(monkeypatch, tmp_path):
    captured_memory_ids = []
    fake_llm = _FakeLLM()
    predicted_probs = {
        "m-1": 0.9,
        "m-2": 0.9,
        "m-3": 0.9,
    }

    def fake_agentic(question, llm, search_engine, **kwargs):
        captured_memory_ids.append(
            [item.source_question_id for item in (kwargs.get("injected_memories") or [])]
        )
        return {
            "predicted_prob": predicted_probs[str(question["market_id"])],
            "reasoning_summary": f"summary {question['market_id']}",
            "retrieved_source_types": ["search"],
            "retrieved_docs": [],
            "trajectory": [{"step": "final", "raw_response": f"trace {question['market_id']}"}],
        }

    monkeypatch.setattr("forecasting.methods.reasoningbank.run_agentic_forecast", fake_agentic)
    monkeypatch.setattr("forecasting.methods.reasoningbank.ResidentCodeInterpreterTool", _FakeCodeInterpreter)
    monkeypatch.setattr(
        "forecasting.methods.reasoningbank.build_text_embedder",
        lambda **kwargs: _FakeEmbedder(),
    )

    session = ReasoningBankMethod().build_session(_runtime_ctx(tmp_path, llm=fake_llm), {"agent_max_steps": 3, "top_k": 1})
    session.run_question(
        _question("m-1", open_time="2026-01-01T00:00:00Z", resolve_time="2026-01-02T00:00:00Z", label=1, domain="finance")
    )
    session.run_question(
        _question("m-2", open_time="2026-01-02T00:00:00Z", resolve_time="2026-01-05T00:00:00Z", label=1, domain="world")
    )
    session.run_question(
        _question(
            "m-3",
            open_time="2026-01-03T00:00:00Z",
            sample_time="2026-01-06T00:00:00Z",
            resolve_time="2026-01-08T00:00:00Z",
            label=1,
            domain="finance",
        )
    )
    artifacts = session.finalize()

    assert captured_memory_ids == [[], [], ["m-1"]]
    assert fake_llm.chat_calls == 3
    assert len(artifacts) == 1
    assert artifacts[0].filename == "reasoningbank_mem.jsonl"
    assert len(artifacts[0].payload) == 3
    assert len(artifacts[0].payload[0]["memory_items"]) == 1
    assert artifacts[0].payload[0]["query_embedding"]


def test_flex_session_preloads_only_resolved_experiences(monkeypatch, tmp_path):
    captured_preloaded = []
    fake_llm = _FakeLLM()
    predicted_probs = {
        "m-1": 0.9,
        "m-2": 0.9,
        "m-3": 0.9,
    }

    def fake_agentic(question, llm, search_engine, **kwargs):
        captured_preloaded.append(
            [
                (item.source_question_id, item.domain, item.zone)
                for item in (kwargs.get("flex_preloaded") or [])
            ]
        )
        return {
            "predicted_prob": predicted_probs[str(question["market_id"])],
            "reasoning_summary": f"summary {question['market_id']}",
            "retrieved_source_types": ["search"],
            "retrieved_docs": [],
            "trajectory": [{"step": "final", "raw_response": f"trace {question['market_id']}"}],
        }

    monkeypatch.setattr("forecasting.methods.flex.run_agentic_forecast", fake_agentic)
    monkeypatch.setattr("forecasting.methods.flex.ResidentCodeInterpreterTool", _FakeCodeInterpreter)
    monkeypatch.setattr(
        "forecasting.methods.flex.build_text_embedder",
        lambda **kwargs: _FakeEmbedder(),
    )

    session = FlexMethod().build_session(
        _runtime_ctx(tmp_path, llm=fake_llm),
        {"agent_max_steps": 3, "strategy_top_k": 1, "pattern_top_k": 1, "case_top_k": 1},
    )
    session.run_question(
        _question("m-1", open_time="2026-01-01T00:00:00Z", resolve_time="2026-01-05T00:00:00Z", label=1, domain="finance")
    )
    session.run_question(
        _question("m-2", open_time="2026-01-03T00:00:00Z", resolve_time="2026-01-04T00:00:00Z", label=1, domain="world")
    )
    session.run_question(
        _question(
            "m-3",
            open_time="2026-01-04T00:00:00Z",
            sample_time="2026-01-06T00:00:00Z",
            resolve_time="2026-01-08T00:00:00Z",
            label=1,
            domain="finance",
        )
    )
    artifacts = session.finalize()

    assert captured_preloaded == [[], [], [("m-1", "finance", "golden")] * 3]
    assert fake_llm.chat_json_calls == 3
    assert len(artifacts) == 1
    assert artifacts[0].filename == "flex_mem.jsonl"
    assert len(artifacts[0].payload) >= 3
    assert artifacts[0].payload[0]["embedding"]
