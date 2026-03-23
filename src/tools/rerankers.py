"""Local rerankers for second-stage passage reordering."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Sequence


DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"
_RERANKER_CACHE: dict[tuple[str, str | None, int, int], "HuggingFaceTextReranker"] = {}


@dataclass
class HuggingFaceTextReranker:
    model_name: str = DEFAULT_RERANKER_MODEL
    device: str | None = None
    batch_size: int = 32
    max_length: int = 512

    _model: Any = None
    _tokenizer: Any = None
    _torch: Any = None
    _device: str | None = None

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        if not documents:
            return []
        self._ensure_loaded()
        outputs: list[float] = []
        normalized_query = str(query or "").strip()
        for start in range(0, len(documents), self.batch_size):
            batch_documents = [str(item or "") for item in documents[start : start + self.batch_size]]
            encoded = self._tokenizer(
                [normalized_query] * len(batch_documents),
                batch_documents,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self._device) for key, value in encoded.items()}
            with self._torch.inference_mode():
                logits = self._model(**encoded).logits
            scores = _extract_relevance_scores(self._torch, logits)
            outputs.extend(float(value) for value in scores.cpu().tolist())
        return outputs

    @property
    def resolved_device(self) -> str:
        return str(self._device or _resolve_torch_device(self.device))

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("transformers and torch are required for local reranking") from exc
        self._torch = torch
        self._device = _resolve_torch_device(self.device)
        try:
            self._tokenizer = _load_from_pretrained(AutoTokenizer, self.model_name)
            self._model = _load_from_pretrained(AutoModelForSequenceClassification, self.model_name)
        except Exception as exc:
            proxy_hint = _active_proxy_summary()
            message = (
                f"Failed to load Hugging Face reranker model {self.model_name!r}: {exc}. "
                "Ensure the model is downloadable in the current environment or pre-cache it locally."
            )
            if proxy_hint:
                message += f" Active proxy env: {proxy_hint}"
            raise RuntimeError(message) from exc
        self._model.eval()
        self._model.to(self._device)


def build_text_reranker(
    *,
    model_name: str = DEFAULT_RERANKER_MODEL,
    device: str | None = None,
    batch_size: int = 32,
    max_length: int = 512,
) -> HuggingFaceTextReranker:
    resolved_batch_size = max(1, int(batch_size or 32))
    resolved_max_length = max(32, int(max_length or 512))
    key = (model_name, device, resolved_batch_size, resolved_max_length)
    cached = _RERANKER_CACHE.get(key)
    if cached is not None:
        return cached
    reranker = HuggingFaceTextReranker(
        model_name=model_name,
        device=device,
        batch_size=resolved_batch_size,
        max_length=resolved_max_length,
    )
    _RERANKER_CACHE[key] = reranker
    return reranker


def _extract_relevance_scores(torch: Any, logits: Any) -> Any:
    if logits.ndim == 0:
        return logits.reshape(1)
    if logits.ndim == 1:
        return logits
    if logits.shape[-1] == 1:
        return logits.squeeze(-1)
    return logits[..., -1]


def _resolve_torch_device(device: str | None) -> str:
    value = str(device or "").strip().lower()
    if value:
        return value
    try:
        import torch
    except Exception:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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
