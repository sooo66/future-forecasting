from __future__ import annotations

from forecasting.datasets.questions import select_question_subset


def test_select_question_subset_is_deterministic_and_balanced():
    questions = []
    domains = ["finance", "politics", "tech", "world"]
    difficulties = ["easy", "medium", "hard"]
    for idx in range(24):
        questions.append(
            {
                "market_id": f"m-{idx:02d}",
                "question": f"Question {idx}",
                "description": "Synthetic question",
                "resolution_criteria": "Synthetic resolution",
                "domain": domains[idx % len(domains)],
                "open_time": f"2026-01-{(idx % 12) + 1:02d}T00:00:00Z",
                "resolve_time": f"2026-02-{(idx % 12) + 1:02d}T00:00:00Z",
                "resolved_time": f"2026-02-{(idx % 12) + 1:02d}T00:00:00Z",
                "sample_time": f"2026-01-{(idx % 12) + 1:02d}T00:00:00Z",
                "difficulty": difficulties[idx % len(difficulties)],
                "sampled_prob_yes": 0.5,
                "label": idx % 2,
                "horizon": "7d",
            }
        )

    subset_a, stats_a = select_question_subset(questions, target_size=12, seed=20260313, time_bucket_count=4)
    subset_b, stats_b = select_question_subset(questions, target_size=12, seed=20260313, time_bucket_count=4)

    assert [item["market_id"] for item in subset_a] == [item["market_id"] for item in subset_b]
    assert stats_a["domains"].keys() == {"finance", "politics", "tech", "world"}
    assert stats_a["difficulties"].keys() == {"easy", "medium", "hard"}
    assert len(stats_b["open_bucket_distribution"]) == 4
    assert len(stats_b["resolve_bucket_distribution"]) == 4
    assert stats_a["open_bucket_specs"][0]["start"] <= stats_a["open_bucket_specs"][1]["start"]
    assert stats_a["resolve_bucket_specs"][0]["start"] <= stats_a["resolve_bucket_specs"][1]["start"]
