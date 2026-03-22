from __future__ import annotations

import json
from types import SimpleNamespace

from forecasting.contracts import MethodArtifact
from forecasting.runner import run_experiment


_RUN_ORDER: list[str] = []


class _FakeLLM:
    def __init__(self, project_root):
        self.model = "fake-model"
        self.base_url = "https://fake.local"


class _FakeSearchClient:
    def __init__(self, *args, **kwargs):
        self.base_url = "http://127.0.0.1:8000"

    def health(self):
        return {"snapshot_root": str((_PROJECT_ROOT / "data" / "mini").resolve())}


class _FakeSession:
    def __init__(self, result_path):
        self.result_path = result_path
        self.calls = 0

    def run_question(self, question):
        if self.result_path.exists():
            existing_lines = [line for line in self.result_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            existing_lines = []
        assert len(existing_lines) == self.calls

        _RUN_ORDER.append(question["market_id"])
        self.calls += 1
        return {
            "market_id": question["market_id"],
            "domain": question["domain"],
            "difficulty": question["difficulty"],
            "open_time": question["open_time"],
            "resolve_time": question["resolve_time"],
            "sample_time": question.get("sample_time"),
            "method_name": "fake_method",
            "predicted_prob": 0.8,
            "label": question["label"],
            "accuracy": 1,
            "brier_score": 0.04,
            "log_loss": 0.1,
            "trajectory": [],
            "retrieved_docs": [],
            "retrieved_memories": [],
            "retrieved_source_types": ["search"],
            "tool_usage_counts": {"search": 1},
            "reasoning_summary": "synthetic",
            "latency_sec": 0.1,
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "steps_count": 4,
        }

    def finalize(self):
        return [MethodArtifact(filename="fake_artifact.jsonl", format="jsonl", payload=[{"id": "artifact-1"}])]


class _FakeMethod:
    method_id = "fake_method"

    def build_session(self, runtime_ctx, method_config=None):
        return _FakeSession(runtime_ctx.output_dir / "results_fake_method.jsonl")


_PROJECT_ROOT = None


def test_runner_executes_spec_and_writes_outputs(monkeypatch, tmp_path):
    global _PROJECT_ROOT
    _PROJECT_ROOT = tmp_path
    _RUN_ORDER.clear()
    questions = [
        {
            "market_id": "m-3",
            "question": "Question 3",
            "description": "Synthetic question",
            "resolution_criteria": "Synthetic resolution",
            "domain": "finance",
            "open_time": "2026-01-03T00:00:00Z",
            "resolve_time": "2026-01-08T00:00:00Z",
            "resolved_time": "2026-01-08T00:00:00Z",
            "sample_time": "2026-01-03T00:00:00Z",
            "difficulty": "easy",
            "sampled_prob_yes": 0.5,
            "label": 1,
            "horizon": "7d",
        },
        {
            "market_id": "m-1",
            "question": "Question 1",
            "description": "Synthetic question",
            "resolution_criteria": "Synthetic resolution",
            "domain": "finance",
            "open_time": "2026-01-02T00:00:00Z",
            "resolve_time": "2026-01-04T00:00:00Z",
            "resolved_time": "2026-01-04T00:00:00Z",
            "sample_time": "2026-01-02T00:00:00Z",
            "difficulty": "easy",
            "sampled_prob_yes": 0.5,
            "label": 1,
            "horizon": "7d",
        },
        {
            "market_id": "m-2",
            "question": "Question 2",
            "description": "Synthetic question",
            "resolution_criteria": "Synthetic resolution",
            "domain": "finance",
            "open_time": "2026-01-01T00:00:00Z",
            "resolve_time": "2026-01-06T00:00:00Z",
            "resolved_time": "2026-01-06T00:00:00Z",
            "sample_time": "2026-01-01T00:00:00Z",
            "difficulty": "easy",
            "sampled_prob_yes": 0.5,
            "label": 1,
            "horizon": "7d",
        },
    ]
    spec = SimpleNamespace(
        experiment_id="fake_exp",
        dataset_file="data/questions/subsets/fake.json",
        knowledge_root="data/mini",
        method_ids=["fake_method"],
        method_configs={"fake_method": {"foo": "bar"}},
        output_dir="artifacts/fake_exp",
        max_parallel_methods=1,
    )

    monkeypatch.setattr("forecasting.runner.OpenAIChatModel", _FakeLLM)
    monkeypatch.setattr("forecasting.runner.SearchClient", _FakeSearchClient)
    monkeypatch.setattr("forecasting.runner.get_method", lambda method_id: _FakeMethod())
    monkeypatch.setattr(
        "forecasting.runner.load_fixed_question_subset",
        lambda project_root, subset_file: {
            "questions": questions,
            "subset_id": "fake_subset",
            "horizon": "7d",
            "stats": {"selected_count": 3},
        },
    )

    result = run_experiment(spec, project_root=tmp_path)

    output_dir = tmp_path / "artifacts" / "fake_exp"
    assert result["output_dir"] == str(output_dir.resolve())
    assert (output_dir / "results_fake_method.jsonl").exists()
    assert (output_dir / "summary.md").exists()
    assert (output_dir / "fake_artifact.jsonl").exists()
    assert not (output_dir / "question_subset_7d.json").exists()
    assert not (output_dir / "question_subset_7d_stats.json").exists()
    assert not (output_dir / "calibration_fake_method.json").exists()
    assert not (tmp_path / "docs" / "fake-exp.md").exists()

    summary_text = (output_dir / "summary.md").read_text(encoding="utf-8")
    result_rows = [json.loads(line) for line in (output_dir / "results_fake_method.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert "# Experiment Summary: fake_exp" in summary_text
    assert "| Method | Count | Accuracy | Brier | ECE@10 | Avg latency (s) | Avg total tokens | Avg steps |" in summary_text
    assert "| fake_method | 3 | 1.000 | 0.0400 | 0.2000 | 0.10 | 15 | 4.0 |" in summary_text
    assert [row["market_id"] for row in result_rows] == ["m-1", "m-2", "m-3"]
    assert _RUN_ORDER == ["m-1", "m-2", "m-3"]
