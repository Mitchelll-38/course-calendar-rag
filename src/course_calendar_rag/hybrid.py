"""Hybrid retrieval using weighted reciprocal rank fusion."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .bm25 import BM25Index, SearchResult
from .dense import DenseIndex, Embedder


def reciprocal_rank_fusion(
    rankings: list[list[SearchResult]], *, weights: list[float] | None = None, k: int = 60
) -> list[SearchResult]:
    if not rankings:
        return []
    weights = weights or [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("one weight is required per ranking")
    scores: dict[str, float] = defaultdict(float)
    passages: dict[str, dict[str, Any]] = {}
    for ranking, weight in zip(rankings, weights):
        for rank, result in enumerate(ranking, start=1):
            scores[result.passage_id] += weight / (k + rank)
            passages[result.passage_id] = result.passage
    fused = [SearchResult(item_id, score, passages[item_id]) for item_id, score in scores.items()]
    fused.sort(key=lambda result: (-result.score, result.passage_id))
    return fused


class HybridIndex:
    def __init__(
        self,
        passages: list[dict[str, Any]],
        *,
        embedder: Embedder | None = None,
        lexical_weight: float = 1.0,
        dense_weight: float = 1.0,
        rrf_k: int = 60,
        candidate_k: int = 10,
    ) -> None:
        self.bm25 = BM25Index(passages)
        self.dense = DenseIndex(passages, embedder)
        self.weights = [lexical_weight, dense_weight]
        self.rrf_k = rrf_k
        self.candidate_k = candidate_k

    def search(self, query: str, *, top_k: int = 5) -> list[SearchResult]:
        candidate_k = max(top_k, self.candidate_k)
        fused = reciprocal_rank_fusion(
            [
                self.bm25.search(query, top_k=candidate_k),
                self.dense.search(query, top_k=candidate_k),
            ],
            weights=self.weights,
            k=self.rrf_k,
        )
        return fused[:top_k]
