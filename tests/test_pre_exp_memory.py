from __future__ import annotations

from forecasting.memory import FlexExperience, FlexLibrary, MemoryItem, ReasoningBankStore


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


def test_reasoningbank_store_flat_item_add_and_retrieve():
    """Per the paper: simple addition of items, retrieval by cosine similarity on embedding."""
    store = ReasoningBankStore(embedder=_FakeEmbedder(), model_name="fake-embedder")

    # Add items from two different "experiences"
    store.add_items([
        MemoryItem(title="AAPL signal", description="structured", content="use structured evidence first"),
    ])
    store.add_items([
        MemoryItem(title="Avoid weak", description="warning", content="do not overfit weak trends"),
    ])

    # Retrieve top-1 matching items
    hits = store.retrieve("AAPL structured signal", top_k=1)

    assert len(hits) == 1
    assert hits[0].title == "AAPL signal"
    assert hits[0].content == "use structured evidence first"


def test_reasoningbank_store_returns_top_k_items():
    """Retrieve top_k most similar items, per the paper's top_k design."""
    store = ReasoningBankStore(embedder=_FakeEmbedder(), model_name="fake-embedder")

    store.add_items([
        MemoryItem(title="AAPL cue", description="finance", content="AAPL financial signal"),
    ])
    store.add_items([
        MemoryItem(title="World cue", description="world", content="World news signal"),
    ])

    hits = store.retrieve("AAPL stock signal", top_k=2)

    assert len(hits) == 2
    assert hits[0].title == "AAPL cue"
    assert hits[1].title == "World cue"


def test_reasoningbank_store_returns_empty_when_no_items():
    """Empty store returns empty list."""
    store = ReasoningBankStore(embedder=_FakeEmbedder(), model_name="fake-embedder")
    hits = store.retrieve("AAPL signal", top_k=1)
    assert hits == []


def test_reasoningbank_store_artifact_rows():
    """artifact_rows exports all items with their embeddings."""
    store = ReasoningBankStore(embedder=_FakeEmbedder(), model_name="test-model")
    store.add_items([
        MemoryItem(title="Test title", description="test desc", content="test content"),
    ])

    rows = store.artifact_rows()
    assert len(rows) == 1
    assert rows[0]["title"] == "Test title"
    assert rows[0]["embedding_model_name"] == "test-model"
    assert len(rows[0]["embedding"]) > 0


def test_flex_library_merges_similar_entries_and_retrieves_hierarchical_bundle():
    library = FlexLibrary(embedder=_FakeEmbedder(), model_name="fake-embedder", merge_similarity_threshold=0.8)
    library.queue_many(
        [
            FlexExperience(
                experience_id="exp-strategy-1",
                source_question_id="q1",
                domain="tech",
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
                domain="tech",
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
                experience_id="exp-strategy-other-domain",
                source_question_id="q-other",
                domain="world",
                zone="golden",
                level="strategy",
                title="Use structured evidence first",
                summary="same title different domain",
                content="similar guidance but different domain",
                created_at="2026-01-04T12:00:00Z",
                source_open_time="2026-01-02T00:00:00Z",
                source_resolved_time="2026-01-04T12:00:00Z",
                outcome=1,
                correctness=True,
            ),
            FlexExperience(
                experience_id="exp-pattern-q3",
                source_question_id="q3",
                domain="world",
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
                domain="finance",
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
                domain="finance",
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
                domain="finance",
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
        zone="golden",
        domain="finance",
    )

    assert len(merged_strategy) == 2
    assert sorted(item.domain for item in merged_strategy) == ["tech", "world"]
    tech_strategy = next(item for item in merged_strategy if item.domain == "tech")
    assert tech_strategy.support_count == 2
    assert [item.source_question_id for item in bundle] == ["q4", "q4", "q4"]
