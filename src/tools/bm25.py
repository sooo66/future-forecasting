"""Shared BM25 helpers for search and memory retrieval."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re
from typing import Any, Optional


_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")
_BM25_K1 = 1.5
_BM25_B = 0.75


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in _TOKEN_PATTERN.finditer(text or "")]


@dataclass(frozen=True)
class BM25Document:
    doc_id: str
    text: str
    metadata: dict[str, Any]


class BM25Index:
    def __init__(self, documents: Optional[list[BM25Document]] = None) -> None:
        self.documents: list[BM25Document] = []
        self._doc_term_frequencies: list[dict[str, int]] = []
        self._doc_lengths: list[int] = []
        self._document_frequencies: dict[str, int] = {}
        self._avg_doc_length = 0.0
        if documents:
            self.reset(documents)

    def reset(self, documents: list[BM25Document]) -> None:
        self.documents = list(documents)
        self._doc_term_frequencies = []
        self._doc_lengths = []
        self._document_frequencies = {}
        self._avg_doc_length = 0.0
        if not self.documents:
            return

        document_frequencies: Counter[str] = Counter()
        total_doc_length = 0
        for document in self.documents:
            tokens = tokenize(document.text)
            total_doc_length += len(tokens)
            term_frequencies = Counter(tokens)
            self._doc_term_frequencies.append(dict(term_frequencies))
            self._doc_lengths.append(len(tokens))
            document_frequencies.update(term_frequencies.keys())

        self._document_frequencies = dict(document_frequencies)
        self._avg_doc_length = total_doc_length / max(len(self.documents), 1)

    def scores(self, query: str, *, candidate_indices: Optional[list[int]] = None) -> list[float]:
        if candidate_indices is None:
            candidate_indices = list(range(len(self.documents)))
        return self._bm25_scores(query, candidate_indices)

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        candidate_indices: Optional[list[int]] = None,
    ) -> list[tuple[BM25Document, float]]:
        if not self.documents:
            return []
        if candidate_indices is None:
            candidate_indices = list(range(len(self.documents)))
        scores = self._bm25_scores(query, candidate_indices)
        ranked = sorted(zip(candidate_indices, scores), key=lambda item: item[1], reverse=True)
        out: list[tuple[BM25Document, float]] = []
        for index, score in ranked[: max(1, top_k)]:
            if score <= 0:
                continue
            out.append((self.documents[index], score))
        return out

    def _bm25_scores(self, query: str, candidate_indices: list[int]) -> list[float]:
        query_tokens = tokenize(query)
        if not query_tokens or not self.documents:
            return [0.0 for _ in candidate_indices]

        total_docs = len(self.documents)
        avg_doc_length = self._avg_doc_length or 1.0
        scores: list[float] = []
        for index in candidate_indices:
            term_frequencies = self._doc_term_frequencies[index]
            doc_length = self._doc_lengths[index]
            score = 0.0
            for token in query_tokens:
                tf = term_frequencies.get(token, 0)
                if tf <= 0:
                    continue
                doc_freq = self._document_frequencies.get(token, 0)
                if doc_freq <= 0:
                    continue
                idf = math.log(1.0 + ((total_docs - doc_freq + 0.5) / (doc_freq + 0.5)))
                denom = tf + _BM25_K1 * (1.0 - _BM25_B + _BM25_B * (doc_length / avg_doc_length))
                score += idf * ((tf * (_BM25_K1 + 1.0)) / denom)
            scores.append(score)
        return scores
