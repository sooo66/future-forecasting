from __future__ import annotations

from experiments import get_experiment_spec


def test_pre_exp_spec_defaults_to_expected_methods():
    spec = get_experiment_spec("pre_experiment")
    assert spec.method_ids == ["direct_io", "bm25_rag", "agentic_nomem", "reasoningbank", "flex"]
    assert spec.dataset_file == "data/questions/subsets/pre_exp_fixed_30_resolved.json"
    assert spec.output_dir == "artifacts/pre_experiment"
    assert "agentic_mem" not in spec.method_configs
    assert spec.method_configs["reasoningbank"]["top_k"] == 1


def test_pre_exp_smoke_v3_build_function_writes_to_v3_output_dir():
    spec = get_experiment_spec("pre_experiment_smoke_v3")
    assert spec.experiment_id == "pre_experiment_smoke_v3"
    assert spec.output_dir == "artifacts/pre_experiment_smoke_v3"


def test_pre_exp_smoke_3_build_function_uses_fixed_three_question_subset():
    spec = get_experiment_spec("pre_experiment_smoke_3")
    assert spec.experiment_id == "pre_experiment_smoke_3"
    assert spec.output_dir == "artifacts/pre_experiment_smoke_3"
    assert spec.dataset_file == "data/questions/subsets/pre_exp_smoke_3_resolved.json"
