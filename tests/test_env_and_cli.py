from __future__ import annotations

import os
import json
from argparse import Namespace
from types import SimpleNamespace

from cli import _build_llm_config, _list_experiment_specs, _parse_methods_override, _resolve_agent_prompt, _resolve_search_api_base, _resolve_snapshot_root, _run_experiment_command, _run_experiment_pretty_command
from utils.env import load_dotenv


def test_load_dotenv_keeps_existing_env_by_default(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "MODEL_NAME=from-file\nV_API_BASE_URL=https://api.example.com/v1\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("MODEL_NAME", "from-env")
    monkeypatch.delenv("V_API_BASE_URL", raising=False)

    loaded = load_dotenv(env_path, override=False)

    assert loaded["MODEL_NAME"] == "from-file"
    assert os.getenv("MODEL_NAME") == "from-env"
    assert os.getenv("V_API_BASE_URL") == "https://api.example.com/v1"


def test_build_llm_config_supports_project_env_aliases(monkeypatch):
    for key in [
        "MODEL_NAME",
        "QWEN_MODEL",
        "LLM_MODEL",
        "OPENAI_MODEL",
        "V_API_BASE_URL",
        "QWEN_MODEL_SERVER",
        "OPENAI_BASE_URL",
        "BASE_URL",
        "LLM_BASE_URL",
        "MODEL_SERVER",
        "V_API_KEY",
        "QWEN_API_KEY",
        "DASHSCOPE_API_KEY",
        "OPENAI_API_KEY",
        "API_KEY",
        "LLM_API_KEY",
    ]:
        monkeypatch.delenv(key, raising=False)

    monkeypatch.setenv("MODEL_NAME", "qwen-plus")
    monkeypatch.setenv("V_API_BASE_URL", "https://api.example.com/v1")
    monkeypatch.setenv("V_API_KEY", "secret")

    cfg = _build_llm_config(Namespace(model="", model_server="", api_key=""))

    assert cfg == {
        "model": "qwen-plus",
        "model_server": "https://api.example.com/v1",
        "api_key": "secret",
    }


def test_agent_prompt_and_search_api_resolution(monkeypatch):
    monkeypatch.setenv("SEARCH_API_BASE", "http://127.0.0.1:9000")

    args = Namespace(
        prompt="flag prompt",
        prompt_text=["positional", "prompt"],
        search_api_base="",
    )
    assert _resolve_agent_prompt(args) == "flag prompt"
    assert _resolve_search_api_base(args) == "http://127.0.0.1:9000"

    args = Namespace(
        prompt="",
        prompt_text=["positional", "prompt"],
        search_api_base="http://127.0.0.1:9100",
    )
    assert _resolve_agent_prompt(args) == "positional prompt"
    assert _resolve_search_api_base(args) == "http://127.0.0.1:9100"


def test_snapshot_root_resolution_for_search_serve(monkeypatch):
    monkeypatch.setenv("SNAPSHOT_ROOT", "data/benchmark/from-env")
    args = Namespace(snapshot_root="")
    assert _resolve_snapshot_root(args) == "data/benchmark/from-env"


def test_parse_methods_override():
    assert _parse_methods_override("") == []
    assert _parse_methods_override("all") == []
    assert _parse_methods_override("direct_io, flex") == ["direct_io", "flex"]


def test_list_experiment_specs(monkeypatch):
    monkeypatch.setattr("cli.list_experiments", lambda: ["pre_experiment", "pre_experiment_smoke"])
    assert _list_experiment_specs() == ["pre_experiment", "pre_experiment_smoke"]


def test_run_experiment_command_uses_spec_and_overrides(monkeypatch, tmp_path):
    captured = {}
    spec = SimpleNamespace(experiment_id="pre_experiment", output_dir="artifacts/pre_experiment")

    monkeypatch.setattr("cli.get_experiment_spec", lambda experiment_id: spec)
    monkeypatch.setattr("cli.setup_logger", lambda *args, **kwargs: None)

    def fake_run_experiment(spec_obj, **kwargs):
        captured["spec"] = spec_obj
        captured["kwargs"] = kwargs
        return {"output_dir": str(tmp_path / "artifacts" / "pre_experiment")}

    monkeypatch.setattr("cli.run_experiment", fake_run_experiment)

    exit_code = _run_experiment_command(
        Namespace(
            id="pre_experiment",
            output_dir="custom/output",
            methods="direct_io,flex",
            search_api_base="http://127.0.0.1:9001",
            force=True,
            max_parallel_methods=3,
            verbose=False,
        )
    )

    assert exit_code == 0
    assert captured["spec"] is spec
    assert captured["kwargs"]["methods_override"] == ["direct_io", "flex"]
    assert captured["kwargs"]["output_dir_override"] == "custom/output"
    assert captured["kwargs"]["search_api_base"] == "http://127.0.0.1:9001"
    assert captured["kwargs"]["force"] is True
    assert captured["kwargs"]["max_parallel_methods"] == 3


def test_run_experiment_pretty_command_renders_jsonl(capsys, tmp_path):
    result_path = tmp_path / "results_fake.jsonl"
    rows = [
        {"market_id": "m-1", "predicted_prob": 0.7},
        {"market_id": "m-2", "predicted_prob": 0.2},
    ]
    result_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

    exit_code = _run_experiment_pretty_command(Namespace(file=str(result_path), output=""))

    assert exit_code == 0
    rendered = capsys.readouterr().out
    assert rendered.startswith("[\n  {\n")
    assert '"market_id": "m-1"' in rendered
    assert '"predicted_prob": 0.2' in rendered


def test_run_experiment_pretty_command_writes_output_file(tmp_path):
    result_path = tmp_path / "results_fake.jsonl"
    output_path = tmp_path / "pretty.json"
    result_path.write_text('{"market_id":"m-1"}\n', encoding="utf-8")

    exit_code = _run_experiment_pretty_command(Namespace(file=str(result_path), output=str(output_path)))

    assert exit_code == 0
    assert output_path.exists()
    assert json.loads(output_path.read_text(encoding="utf-8")) == [{"market_id": "m-1"}]
