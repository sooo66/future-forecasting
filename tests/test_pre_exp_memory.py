from __future__ import annotations

from forecasting.memory import FlexExperience, FlexLibrary, MemoryItem, ReasoningBankRecord, ReasoningBankStore


class _FakeEmbedder:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            values = [
                1.0 if "aapl" in lowered or "apple" in lowered else 0.0,
                1.0 if "warning" in lowered or "avoid" in lowered else 0.0,
                1.0 if "strategy" in lowered or "structured" in lowered else 0.0,
            ]
            norm = sum(value * value for value in values) ** 0.5 or 1.0
            vectors.append([value / norm for value in values])
        return vectors


def test_reasoningbank_store_only_activates_past_resolved_records():
    store = ReasoningBankStore(embedder=_FakeEmbedder(), model_name="fake-embedder")
    store.queue(
        ReasoningBankRecord(
            record_id="rb-past",
            source_question_id="q1",
            query="AAPL supply signal",
            trajectory=[{"step": "final", "raw_response": "past"}],
            memory_items=[MemoryItem(title="Past cue", description="usable", content="useful signal")],
            created_at="2026-01-05T00:00:00Z",
            source_open_time="2026-01-01T00:00:00Z",
            source_resolved_time="2026-01-05T00:00:00Z",
            outcome=1,
            predicted_prob=0.9,
            success_or_failure="success",
        )
    )
    store.queue(
        ReasoningBankRecord(
            record_id="rb-future",
            source_question_id="q2",
            query="AAPL future warning",
            trajectory=[{"step": "final", "raw_response": "future"}],
            memory_items=[MemoryItem(title="Future cue", description="not yet active", content="should not leak")],
            created_at="2026-02-05T00:00:00Z",
            source_open_time="2026-02-01T00:00:00Z",
            source_resolved_time="2026-02-05T00:00:00Z",
            outcome=0,
            predicted_prob=0.1,
            success_or_failure="failure",
        )
    )

    hits = store.retrieve("AAPL signal", open_time="2026-01-20T00:00:00Z", top_k=1)

    assert [(item.record_id, item.title) for item in hits] == [("rb-past", "Past cue")]
    assert hits[0].memory_id == "rb-past#1"


def test_flex_library_merges_similar_entries_and_retrieves_hierarchical_bundle():
    library = FlexLibrary(embedder=_FakeEmbedder(), model_name="fake-embedder", merge_similarity_threshold=0.8)
    library.queue_many(
        [
            FlexExperience(
                experience_id="exp-strategy-1",
                source_question_id="q1",
                zone="golden",
                level="strategy",
                title="Use structured evidence first",
                summary="prefer structured data",
                content="use structured data first",
                created_at="2026-01-03T00:00:00Z",
                source_open_time="2026-01-01T00:00:00Z",
                source_resolved_time="2026-01-03T00:00:00Z",
                outcome=1,
                correctness=True,
            ),
            FlexExperience(
                experience_id="exp-strategy-2",
                source_question_id="q2",
                zone="golden",
                level="strategy",
                title="Use structured evidence first",
                summary="prefer structured data strongly",
                content="use structured data before generic news",
                created_at="2026-01-04T00:00:00Z",
                source_open_time="2026-01-02T00:00:00Z",
                source_resolved_time="2026-01-04T00:00:00Z",
                outcome=1,
                correctness=True,
            ),
        ]
    )
    library.queue_many(
        [
            FlexExperience(
                experience_id="exp-pattern-q3",
                source_question_id="q3",
                zone="warning",
                level="pattern",
                title="Avoid unsupported extrapolation",
                summary="warning pattern",
                content="avoid unsupported extrapolation",
                created_at="2026-01-05T00:00:00Z",
                source_open_time="2026-01-03T00:00:00Z",
                source_resolved_time="2026-01-05T00:00:00Z",
                outcome=0,
                correctness=False,
            ),
            FlexExperience(
                experience_id="exp-strategy-q4",
                source_question_id="q4",
                zone="golden",
                level="strategy",
                title="AAPL structured strategy",
                summary="AAPL strategy summary",
                content="structured AAPL strategy",
                created_at="2026-01-06T00:00:00Z",
                source_open_time="2026-01-04T00:00:00Z",
                source_resolved_time="2026-01-06T00:00:00Z",
                outcome=1,
                correctness=True,
            ),
            FlexExperience(
                experience_id="exp-pattern-q4",
                source_question_id="q4",
                zone="golden",
                level="pattern",
                title="AAPL structured pattern",
                summary="AAPL pattern summary",
                content="structured AAPL pattern",
                created_at="2026-01-06T00:00:00Z",
                source_open_time="2026-01-04T00:00:00Z",
                source_resolved_time="2026-01-06T00:00:00Z",
                outcome=1,
                correctness=True,
            ),
            FlexExperience(
                experience_id="exp-case-q4",
                source_question_id="q4",
                zone="golden",
                level="case",
                title="AAPL structured case",
                summary="AAPL case summary",
                content="structured AAPL case",
                created_at="2026-01-06T00:00:00Z",
                source_open_time="2026-01-04T00:00:00Z",
                source_resolved_time="2026-01-06T00:00:00Z",
                outcome=1,
                correctness=True,
            ),
        ]
    )

    library.activate_until("2026-01-10T00:00:00Z")
    items = library.items()
    merged_strategy = [item for item in items if item.level == "strategy" and item.title == "Use structured evidence first"]
    bundle = library.retrieve_default_bundle(
        "AAPL structured signal",
        open_time="2026-01-10T00:00:00Z",
        per_level={"strategy": 1, "pattern": 1, "case": 1},
    )

    assert len(merged_strategy) == 1
    assert merged_strategy[0].support_count == 2
    assert [item.source_question_id for item in bundle] == ["q4", "q4", "q4"]
