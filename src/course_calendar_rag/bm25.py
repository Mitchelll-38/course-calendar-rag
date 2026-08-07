"""Small dependency-free BM25 implementation for the lexical baseline."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Any, Iterable

from .tokenize import tokenize


@dataclass(frozen=True)
class SearchResult:
    passage_id: str
    score: float
    passage: dict[str, Any]


class BM25Index:
    def __init__(
        self,
        passages: Iterable[dict[str, Any]],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.passages = list(passages)
        if not self.passages:
            raise ValueError("at least one passage is required")
        if len({passage["id"] for passage in self.passages}) != len(self.passages):
            raise ValueError("passage IDs must be unique")

        self.k1 = k1
        self.b = b
        self.documents = [
            tokenize(f"{passage['title']} {passage['text']}") for passage in self.passages
        ]
        self.term_frequencies = [Counter(document) for document in self.documents]
        self.document_lengths = [len(document) for document in self.documents]
        self.average_document_length = sum(self.document_lengths) / len(self.documents)

        document_frequency: Counter[str] = Counter()
        for document in self.documents:
            document_frequency.update(set(document))
        document_count = len(self.documents)
        self.inverse_document_frequency = {
            term: math.log(1 + (document_count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def search(self, query: str, *, top_k: int = 5) -> list[SearchResult]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        query_terms = tokenize(query)
        scored: list[SearchResult] = []

        for passage, frequencies, document_length in zip(
            self.passages, self.term_frequencies, self.document_lengths
        ):
            score = 0.0
            length_normalization = 1 - self.b + self.b * (
                document_length / self.average_document_length
            )
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                numerator = frequency * (self.k1 + 1)
                denominator = frequency + self.k1 * length_normalization
                score += self.inverse_document_frequency.get(term, 0.0) * numerator / denominator
            scored.append(SearchResult(passage["id"], score, passage))

        scored.sort(key=lambda result: (-result.score, result.passage_id))
        return scored[: min(top_k, len(scored))]

