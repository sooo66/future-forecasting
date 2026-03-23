from __future__ import annotations

from forecasting import embeddings as forecasting_embeddings
from tools import dense_embeddings as search_dense_embeddings


def test_forecasting_embedder_builder_reuses_instances():
    forecasting_embeddings._EMBEDDER_CACHE.clear()

    first = forecasting_embeddings.build_text_embedder()
    second = forecasting_embeddings.build_text_embedder()

    assert first is second


def test_search_dense_embedder_builder_reuses_instances():
    search_dense_embeddings._EMBEDDER_CACHE.clear()

    first = search_dense_embeddings.build_text_embedder()
    second = search_dense_embeddings.build_text_embedder()

    assert first is second
