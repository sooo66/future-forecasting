from __future__ import annotations

from types import SimpleNamespace

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


def test_search_dense_embedder_falls_back_to_cpu_on_cuda_error(monkeypatch):
    class _DummyModel:
        def __init__(self) -> None:
            self.devices: list[str] = []

        def to(self, device: str) -> None:
            self.devices.append(device)

    class _DummyCuda:
        @staticmethod
        def empty_cache() -> None:
            return

    embedder = search_dense_embeddings.HuggingFaceTextEmbedder(device="cuda")
    embedder._model = _DummyModel()
    embedder._torch = SimpleNamespace(cuda=_DummyCuda())
    embedder._device = "cuda"
    monkeypatch.setattr(embedder, "_ensure_loaded", lambda: None)

    calls: list[str] = []

    def fake_embed_texts_impl(texts):
        calls.append(embedder._device or "")
        if embedder._device == "cuda":
            raise RuntimeError("CUDA unknown error")
        return [[1.0]]

    monkeypatch.setattr(embedder, "_embed_texts_impl", fake_embed_texts_impl)

    result = embedder.embed_texts(["hello"])

    assert result == [[1.0]]
    assert calls == ["cuda", "cpu"]
    assert embedder.resolved_device == "cpu"
