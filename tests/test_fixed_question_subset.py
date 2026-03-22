from __future__ import annotations

from collections import Counter
from pathlib import Path

from forecasting.datasets import load_fixed_question_subset


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_fixed_question_subset_loads_balanced_questions():
    loaded = load_fixed_question_subset(
        PROJECT_ROOT,
        "data/questions/subsets/pre_exp_fixed_30_resolved.json",
    )

    assert len(loaded["questions"]) == 30
    assert loaded["subset_id"] == "pre_exp_fixed_30_resolved"
    assert loaded["market_ids"][0] == "898220"
    assert Counter(question["domain"] for question in loaded["questions"]) == {
        "culture": 5,
        "finance": 5,
        "politics": 5,
        "sports": 5,
        "tech": 5,
        "world": 5,
    }
    assert Counter(question["difficulty"] for question in loaded["questions"]) == {
        "easy": 10,
        "hard": 10,
        "medium": 10,
    }
    assert loaded["stats"]["sampling"]["strategy"] == "fixed subset manifest"
    assert loaded["market_ids"] == [
        question["market_id"]
        for question in sorted(
            loaded["questions"],
            key=lambda question: (question["resolve_time"], question["open_time"], question["market_id"]),
        )
    ]
