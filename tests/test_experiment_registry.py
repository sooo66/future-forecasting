from __future__ import annotations

from experiments import get_experiment_spec, list_experiments


def test_experiment_registry_lists_build_function_ids():
    assert list_experiments() == [
        "pre_experiment",
        "smoke_test_3",
        "smoke_test_30",
    ]


def test_experiment_registry_builds_pre_exp_spec():
    spec = get_experiment_spec("pre_experiment")
    assert spec.experiment_id == "pre_experiment"
    assert spec.dataset_file == "data/questions/subsets/pre_exp_fixed_30_resolved.json"
    assert spec.output_dir == "artifacts/pre_experiment"
    assert spec.method_configs["naive_rag"]["search_top_k"] == 3
    assert spec.method_configs["naive_rag"]["search_content_chars"] == 1024
