"""Dense text embeddings for forecasting memory retrieval."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_EMBEDDER_CACHE: dict[tuple[str, str | None], "HuggingFaceTextEmbedder"] = {}


@dataclass
class HuggingFaceTextEmbedder:
    model_name: str = DEFAULT_EMBEDDING_MODEL
    device: str | None = None
    batch_size: int = 16

    _model: Any = None
    _tokenizer: Any = None
    _torch: Any = None

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        self._ensure_loaded()
        outputs: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {key: value.to(self._device) for key, value in encoded.items()}
            with self._torch.no_grad():
                model_output = self._model(**encoded)
            attention_mask = encoded["attention_mask"]
            token_embeddings = model_output.last_hidden_state
            pooled = _mean_pool(self._torch, token_embeddings, attention_mask)
            normalized = self._torch.nn.functional.normalize(pooled, p=2, dim=1)
            outputs.extend(normalized.cpu().tolist())
        return outputs

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("transformers and torch are required for dense embedding retrieval") from exc
        self._torch = torch
        if self.device:
            self._device = self.device
        else:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self._tokenizer = _load_from_pretrained(AutoTokenizer, self.model_name)
            self._model = _load_from_pretrained(AutoModel, self.model_name)
        except Exception as exc:
            proxy_hint = _active_proxy_summary()
            message = (
                f"Failed to load Hugging Face embedding model {self.model_name!r}: {exc}. "
                "Ensure the model is downloadable in the current environment or pre-cache it locally."
            )
            if proxy_hint:
                message += f" Active proxy env: {proxy_hint}"
            raise RuntimeError(message) from exc
        self._model.eval()
        self._model.to(self._device)


def build_text_embedder(
    *,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    device: str | None = None,
) -> HuggingFaceTextEmbedder:
    key = (model_name, device)
    cached = _EMBEDDER_CACHE.get(key)
    if cached is not None:
        return cached
    embedder = HuggingFaceTextEmbedder(model_name=model_name, device=device)
    _EMBEDDER_CACHE[key] = embedder
    return embedder


def _mean_pool(torch: Any, token_embeddings: Any, attention_mask: Any) -> Any:
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = (token_embeddings * mask).sum(dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-12)
    return summed / counts


def _load_from_pretrained(factory: Any, model_name: str) -> Any:
    try:
        return factory.from_pretrained(model_name, local_files_only=True)
    except Exception:
        return factory.from_pretrained(model_name)


def _active_proxy_summary() -> str:
    entries = []
    for key in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        value = os.getenv(key, "").strip()
        if value:
            entries.append(f"{key}={value}")
    return ", ".join(entries)
