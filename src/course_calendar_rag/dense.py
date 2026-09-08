"""Dense retrieval primitives with a deterministic local baseline embedder."""

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
    """Dependency-free feature hashing used for reproducible local evaluation."""

    dimensions: int = 384

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


class SentenceTransformerEmbedder:
    """Optional production adapter; install the `dense` project extra to use it."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("install with `pip install -e .[dense]`") from exc
        self.model = SentenceTransformer(model_name)
        self.dimensions = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()


def cosine_similarity(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


class DenseIndex:
    def __init__(self, passages: Iterable[dict[str, Any]], embedder: Embedder | None = None) -> None:
        self.passages = list(passages)
        if not self.passages:
            raise ValueError("at least one passage is required")
        self.embedder = embedder or HashingEmbedder()
        texts = [f"{item['title']} {item['text']}" for item in self.passages]
        self.vectors = self.embedder.encode(texts)

    def search(self, query: str, *, top_k: int = 5) -> list[SearchResult]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        query_vector = self.embedder.encode([query])[0]
        results = [
            SearchResult(item["id"], cosine_similarity(query_vector, vector), item)
            for item, vector in zip(self.passages, self.vectors)
        ]
        results.sort(key=lambda result: (-result.score, result.passage_id))
        return results[: min(top_k, len(results))]
