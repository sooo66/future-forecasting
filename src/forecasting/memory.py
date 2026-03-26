"""Memory stores for forecasting experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol


class TextEmbedder(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...


@dataclass(frozen=True)
class MemoryItem:
    """Paper-style ReasoningBank memory item."""

    title: str
    description: str
    content: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FlexExperience:
    experience_id: str
    source_question_id: str
    domain: str
    zone: str
    level: str
    title: str
    summary: str
    content: str
    created_at: str
    source_open_time: str
    source_resolved_time: str
    outcome: int
    correctness: bool
    support_count: int = 1
    merged_from: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_tool_dict(self) -> dict[str, Any]:
        return {
            "experience_id": self.experience_id,
            "source_question_id": self.source_question_id,
            "domain": self.domain,
            "zone": self.zone,
            "level": self.level,
            "title": self.title,
            "summary": self.summary,
            "content": self.content,
            "support_count": self.support_count,
            "source_resolved_time": self.source_resolved_time,
            "correctness": self.correctness,
        }


class ReasoningBankStore:
    """Simplified flat pool of ReasoningBank memory items, per the paper's design."""

    def __init__(self, *, embedder: TextEmbedder, model_name: str) -> None:
        self._embedder = embedder
        self._model_name = model_name
        self._items: list[MemoryItem] = []
        self._embeddings: list[list[float]] = []

    def add_items(self, items: list[MemoryItem]) -> None:
        """Simple addition operation: add items directly to the pool with their embeddings."""
        for item in items:
            emb_text = _reasoningbank_embedding_text(item)
            emb = self._embedder.embed_texts([emb_text])[0]
            self._items.append(item)
            self._embeddings.append(emb)

    def retrieve(self, query: str, top_k: int = 1) -> list[MemoryItem]:
        """Retrieve top_k most similar memory items by cosine similarity on embedding."""
        if not self._items:
            return []
        query_emb = self._embedder.embed_texts([query])[0]
        scored = [
            (i, _cosine_similarity(query_emb, emb))
            for i, emb in enumerate(self._embeddings)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [self._items[i] for i, _ in scored[: max(1, top_k)]]

    def items(self) -> list[MemoryItem]:
        return list(self._items)

    def artifact_rows(self) -> list[dict[str, Any]]:
        return [
            {
                **item.to_dict(),
                "embedding_model_name": self._model_name,
                "embedding": emb,
            }
            for item, emb in zip(self._items, self._embeddings)
        ]


def _reasoningbank_embedding_text(item: MemoryItem) -> str:
    """Build the text used for embedding a memory item (title + content)."""
    return f"{item.title} {item.content}"


class FlexLibrary:
    def __init__(
        self,
        *,
        embedder: TextEmbedder,
        model_name: str,
        merge_similarity_threshold: float = 0.92,
    ) -> None:
        self._embedder = embedder
        self._model_name = model_name
        self._merge_similarity_threshold = float(merge_similarity_threshold)
        self._active_items: list[FlexExperience] = []
        self._pending_items: list[tuple[str, FlexExperience]] = []
        self._embedding_cache: dict[str, list[float]] = {}

    def queue_many(self, items: list[FlexExperience]) -> None:
        for item in items:
            self._pending_items.append((item.source_resolved_time, item))
        self._pending_items.sort(key=lambda pair: (pair[0], pair[1].experience_id))

    def activate_until(self, open_time: str) -> None:
        while self._pending_items and self._pending_items[0][0] <= open_time:
            _, item = self._pending_items.pop(0)
            self._insert_or_merge(item)

    def retrieve(
        self,
        query: str,
        *,
        open_time: str,
        top_k: int = 5,
        zone: str | None = None,
        level: str | None = None,
        domain: str | None = None,
        candidate_question_ids: set[str] | None = None,
    ) -> list[FlexExperience]:
        self.activate_until(open_time)
        candidates = [
            item
            for item in self._active_items
            if (not zone or item.zone == zone)
            and (not level or item.level == level)
            and (not domain or item.domain == domain)
            and (candidate_question_ids is None or item.source_question_id in candidate_question_ids)
        ]
        if not candidates:
            return []
        query_embedding = self._embedder.embed_texts([query])[0]
        scored: list[tuple[float, FlexExperience]] = []
        for item in candidates:
            score = _cosine_similarity(query_embedding, self._embedding_cache[item.experience_id])
            scored.append((score, item))
        scored.sort(
            key=lambda pair: (pair[0], pair[1].support_count, pair[1].source_resolved_time, pair[1].experience_id),
            reverse=True,
        )
        return [item for _score, item in scored[: max(1, top_k)]]

    def retrieve_default_bundle(
        self,
        query: str,
        *,
        open_time: str,
        per_level: dict[str, int] | None = None,
        zone: str | None = None,
        domain: str | None = None,
    ) -> list[FlexExperience]:
        level_targets = per_level or {"strategy": 5, "pattern": 5, "case": 5}
        collected: list[FlexExperience] = []
        seen_ids: set[str] = set()
        # Retrieve all three levels independently (per original FLEX paper)
        for level in ("strategy", "pattern", "case"):
            items = self.retrieve(
                query,
                open_time=open_time,
                top_k=level_targets.get(level, 5),
                zone=zone,
                level=level,
                domain=domain,
            )
            for item in items:
                if item.experience_id not in seen_ids:
                    seen_ids.add(item.experience_id)
                    collected.append(item)
        return collected

    def items(self) -> list[FlexExperience]:
        return list(self._active_items)

    def flush_all(self) -> list[FlexExperience]:
        if self._pending_items:
            self.activate_until("9999-12-31T23:59:59Z")
        return self.items()

    def artifact_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for item in self.flush_all():
            payload = item.to_dict()
            payload["embedding_model_name"] = self._model_name
            payload["embedding"] = self._embedding_cache.get(item.experience_id, [])
            rows.append(payload)
        return rows

    def embeddings_payload(self) -> dict[str, Any]:
        ordered = sorted(self._embedding_cache.items())
        return {
            "model_name": self._model_name,
            "records": [{"experience_id": experience_id, "embedding": embedding} for experience_id, embedding in ordered],
        }

    def _insert_or_merge(self, new_item: FlexExperience) -> None:
        new_embedding = self._embed_text(_flex_embedding_text(new_item))
        for index, existing in enumerate(self._active_items):
            if (
                existing.domain != new_item.domain
                or existing.zone != new_item.zone
                or existing.level != new_item.level
            ):
                continue
            if _is_exact_duplicate(existing, new_item):
                return
            score = _cosine_similarity(new_embedding, self._embedding_cache[existing.experience_id])
            if score < self._merge_similarity_threshold:
                continue
            self._active_items[index] = _merge_experiences(existing, new_item)
            self._embedding_cache[existing.experience_id] = self._embed_text(
                _flex_embedding_text(self._active_items[index])
            )
            return
        self._active_items.append(new_item)
        self._embedding_cache[new_item.experience_id] = new_embedding

    def _embed_text(self, text: str) -> list[float]:
        return self._embedder.embed_texts([text])[0]


RollingMemoryStore = ReasoningBankStore


def _flex_embedding_text(item: FlexExperience) -> str:
    return "\n".join(part for part in [item.title, item.summary, item.content] if part)


def _is_exact_duplicate(existing: FlexExperience, new_item: FlexExperience) -> bool:
    return (
        _normalize_text(existing.title) == _normalize_text(new_item.title)
        and _normalize_text(existing.summary) == _normalize_text(new_item.summary)
        and _normalize_text(existing.content) == _normalize_text(new_item.content)
    )


def _merge_experiences(existing: FlexExperience, new_item: FlexExperience) -> FlexExperience:
    return FlexExperience(
        experience_id=existing.experience_id,
        source_question_id=existing.source_question_id,
        domain=existing.domain,
        zone=existing.zone,
        level=existing.level,
        title=_prefer_richer(existing.title, new_item.title),
        summary=_prefer_richer(existing.summary, new_item.summary),
        content=_prefer_richer(existing.content, new_item.content),
        created_at=max(existing.created_at, new_item.created_at),
        source_open_time=min(existing.source_open_time, new_item.source_open_time),
        source_resolved_time=max(existing.source_resolved_time, new_item.source_resolved_time),
        outcome=existing.outcome,
        correctness=existing.correctness,
        support_count=existing.support_count + new_item.support_count,
        merged_from=list(dict.fromkeys(existing.merged_from + [new_item.experience_id])),
    )


def _prefer_richer(left: str, right: str) -> str:
    if len(right.strip()) > len(left.strip()):
        return right.strip()
    return left.strip()


def _normalize_text(text: str) -> str:
    return " ".join((text or "").lower().split())


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return sum(a * b for a, b in zip(left, right))
