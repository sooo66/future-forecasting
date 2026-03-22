"""Experiment specification discovery."""

from __future__ import annotations

from importlib import import_module
from inspect import getmembers, isfunction
from typing import Callable

from experiments.base import ExperimentSpec


def list_experiments() -> list[str]:
    return sorted(_experiment_builders())


def get_experiment_spec(experiment_id: str) -> ExperimentSpec:
    builders = _experiment_builders()
    try:
        return builders[experiment_id]()
    except KeyError as exc:
        raise KeyError(f"Unknown experiment spec: {experiment_id}") from exc


def _experiment_builders() -> dict[str, Callable[[], ExperimentSpec]]:
    module = import_module("experiments.experiment")
    builders: dict[str, Callable[[], ExperimentSpec]] = {}
    for name, value in getmembers(module, isfunction):
        if not name.startswith("build_"):
            continue
        builders[name.removeprefix("build_")] = value
    return builders


__all__ = ["ExperimentSpec", "get_experiment_spec", "list_experiments"]
