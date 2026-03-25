"""Forecasting experiment build functions."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from experiments.base import ExperimentSpec

DEFAULT_METHOD_IDS = [
    "direct_io",
    "naive_rag",
    "agentic_nomem",
    "reasoningbank",
    "flex",
]
DEFAULT_DATASET_FILE = "data/questions/subsets/pre_exp_fixed_30_resolved.json"
DEFAULT_KNOWLEDGE_ROOT = "data/benchmark/s-0306"
DEFAULT_METHOD_CONFIGS = {
    "direct_io": {},
    "naive_rag": {
        "search_top_k": 3,
        "search_content_chars": 1024,
        "rag_max_per_source_type": 10,
    },
    "agentic_nomem": {"agent_max_steps": 8, "search_top_k": 3},
    "reasoningbank": {
        "agent_max_steps": 8,
        "top_k": 1,
        "search_top_k": 3,
        "success_only": True,
        "domain_match": True,
    },
    "flex": {
        "agent_max_steps": 8,
        "search_top_k": 3,
        "strategy_top_k": 3,
        "pattern_top_k": 3,
        "case_top_k": 3,
        "preload_zone": "golden",
        "preload_domain_match": True,
        "memory_tool_domain_match": True,
    },
}


def _build_experiment(
    experiment_id: str,
    *,
    dataset_file: str = DEFAULT_DATASET_FILE,
    knowledge_root: str = DEFAULT_KNOWLEDGE_ROOT,
    method_ids: list[str] | None = None,
    method_configs: dict[str, Any] | None = None,
    max_parallel_methods: int = 1,
) -> ExperimentSpec:
    return ExperimentSpec(
        experiment_id=experiment_id,
        dataset_file=dataset_file,
        knowledge_root=knowledge_root,
        method_ids=list(method_ids or DEFAULT_METHOD_IDS),
        method_configs=deepcopy(method_configs or DEFAULT_METHOD_CONFIGS),
        output_dir=f"artifacts/{experiment_id}",
        max_parallel_methods=max_parallel_methods,
    )


def build_pre_experiment() -> ExperimentSpec:
    return _build_experiment("pre_experiment")


def build_smoke_test_3() -> ExperimentSpec:
    return _build_experiment(
        "smoke_test_3",
        dataset_file="data/questions/subsets/smoke_test_3.json",
    )


def build_smoke_test_30() -> ExperimentSpec:
    return _build_experiment(
        "smoke_test_30",
        dataset_file="data/questions/subsets/smoke_test_30.json",
    )


def build_smoke_test_100() -> ExperimentSpec:
    return _build_experiment(
        "smoke_test_100",
        dataset_file="data/questions/subsets/smoke_test_100.json",
    )


def build_smoke_test_100_gemini() -> ExperimentSpec:
    return _build_experiment(
        "smoke_test_100_gemini",
        method_ids=["reasoningbank"],
        dataset_file="data/questions/subsets/smoke_test_100.json",
    )


def build_smoke_test_30_gemini() -> ExperimentSpec:
    return _build_experiment(
        "smoke_test_30_gemini",
        method_ids=["reasoningbank", "flex"],
        dataset_file="data/questions/subsets/smoke_test_30.json",
    )


def build_smoke_test_30_qwen() -> ExperimentSpec:
    return _build_experiment(
        "smoke_test_30_qwen",
        method_ids=["reasoningbank", "flex"],
        dataset_file="data/questions/subsets/smoke_test_30.json",
    )
