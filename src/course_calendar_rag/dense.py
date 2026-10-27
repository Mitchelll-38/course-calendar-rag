"""Dense retrieval baseline with a dependency-free hashing embedder."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Iterable, Protocol

from .bm25 import SearchResult
from .tokenize import tokenize


class Embedder(Protocol):
    dimensions: int

    def encode(self, texts: list[str]) -> list[list[float]]: ...


@dataclass(frozen=True)
class HashingEmbedder:
    dimensions: int = 384

    def __post_init__(self) -> None:
        if self.dimensions < 1:
            raise ValueError("dimensions must be at least 1")

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [self._encode_one(text) for text in texts]

    def _encode_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        terms = tokenize(text)
        features = terms + [f"{left}_{right}" for left, right in zip(terms, terms[1:])]
        for feature in features:
            digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
            bucket = int.from_bytes(digest, "big") % self.dimensions
            sign = 1.0 if digest[0] & 1 else -1.0
            vector[bucket] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector


def cosine_similarity(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _checked_vectors(
    encoded: Iterable[Iterable[float]], *, count: int, dimensions: int
) -> list[list[float]]:
    try:
        vectors = list(encoded)
    except TypeError as exc:
        raise ValueError("embedder must return a batch of vectors") from exc
    if len(vectors) != count:
        raise ValueError(f"embedder returned {len(vectors)} vectors; expected {count}")
    checked = []
    for vector in vectors:
        try:
            values = list(vector)
        except TypeError as exc:
            raise ValueError("embedder must return vectors of numbers") from exc
        if len(values) != dimensions:
            raise ValueError(
                f"embedder returned a vector with {len(values)} dimensions; "
                f"expected {dimensions}"
            )
        normalized = []
        for value in values:
            if isinstance(value, (bool, str, bytes)):
                raise ValueError("embedder vectors must contain finite numbers")
            try:
                numeric = float(value)
            except (TypeError, ValueError, OverflowError):
                raise ValueError("embedder vectors must contain finite numbers") from None
            if not math.isfinite(numeric):
                raise ValueError("embedder vectors must contain finite numbers")
            normalized.append(numeric)
        checked.append(normalized)
    return checked


class DenseIndex:
    def __init__(
        self, passages: Iterable[dict[str, Any]], *, embedder: Embedder | None = None
    ) -> None:
        self.passages = list(passages)
        if not self.passages:
            raise ValueError("at least one passage is required")
        if len({passage["id"] for passage in self.passages}) != len(self.passages):
            raise ValueError("passage IDs must be unique")
        self.embedder = embedder or HashingEmbedder()
        dimensions = getattr(self.embedder, "dimensions", None)
        if isinstance(dimensions, bool) or not isinstance(dimensions, int) or dimensions < 1:
            raise ValueError("embedder dimensions must be a positive integer")
        self.dimensions = dimensions
        texts = [f"{passage['title']} {passage['text']}" for passage in self.passages]
        self.vectors = _checked_vectors(
            self.embedder.encode(texts), count=len(texts), dimensions=self.dimensions
        )

    def search(self, query: str, *, top_k: int = 5) -> list[SearchResult]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        query_vector = _checked_vectors(
            self.embedder.encode([query]), count=1, dimensions=self.dimensions
        )[0]
        scored = []
        for passage, vector in zip(self.passages, self.vectors):
            score = cosine_similarity(query_vector, vector)
            if not math.isfinite(score):
                raise ValueError("dense similarity score must be finite")
            scored.append(SearchResult(passage["id"], score, passage))
        scored.sort(key=lambda result: (-result.score, result.passage_id))
        return scored[: min(top_k, len(scored))]
