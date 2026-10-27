"""Weighted reciprocal-rank fusion of lexical and dense search results."""

from __future__ import annotations

import math
from typing import Any, Iterable

from .bm25 import BM25Index, SearchResult
from .dense import DenseIndex, Embedder


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _valid_weights(weights: list[float]) -> None:
    if any(
        isinstance(weight, bool)
        or not isinstance(weight, (int, float))
        or not math.isfinite(weight)
        or weight < 0
        for weight in weights
    ):
        raise ValueError("weights must be nonnegative finite numbers")
    if weights and not any(weight > 0 for weight in weights):
        raise ValueError("at least one weight must be positive")


def reciprocal_rank_fusion(
    rankings: list[list[SearchResult]], weights: list[float] | None = None, k: int = 60
) -> list[SearchResult]:
    """Fuse rankings by summing ``weight / (k + rank)`` for each distinct ID."""
    _positive_integer(k, "k")
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights must match the number of rankings")
    _valid_weights(weights)

    scores: dict[str, float] = {}
    passages: dict[str, dict[str, Any]] = {}
    for ranking, weight in zip(rankings, weights):
        if weight == 0:
            continue
        seen: set[str] = set()
        rank = 0
        for result in ranking:
            if result.passage_id in seen:
                continue
            seen.add(result.passage_id)
            rank += 1
            passages.setdefault(result.passage_id, result.passage)
            scores[result.passage_id] = scores.get(result.passage_id, 0.0) + weight / (k + rank)

    return [
        SearchResult(passage_id, score, passages[passage_id])
        for passage_id, score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    ]


class HybridIndex:
    def __init__(
        self,
        passages: Iterable[dict[str, Any]],
        *,
        embedder: Embedder | None = None,
        lexical_weight: float = 1,
        dense_weight: float = 1,
        rrf_k: int = 60,
        candidate_k: int = 10,
    ) -> None:
        _positive_integer(rrf_k, "rrf_k")
        _positive_integer(candidate_k, "candidate_k")
        _valid_weights([lexical_weight, dense_weight])
        self.lexical_weight = lexical_weight
        self.dense_weight = dense_weight
        self.rrf_k = rrf_k
        self.candidate_k = candidate_k
        passage_list = list(passages)
        self.lexical = BM25Index(passage_list) if lexical_weight > 0 else None
        self.dense = DenseIndex(passage_list, embedder=embedder) if dense_weight > 0 else None

    def search(self, query: str, *, top_k: int = 5) -> list[SearchResult]:
        _positive_integer(top_k, "top_k")
        depth = max(top_k, self.candidate_k)
        rankings: list[list[SearchResult]] = []
        weights: list[float] = []
        if self.lexical is not None:
            rankings.append(self.lexical.search(query, top_k=depth))
            weights.append(self.lexical_weight)
        if self.dense is not None:
            rankings.append(self.dense.search(query, top_k=depth))
            weights.append(self.dense_weight)
        return reciprocal_rank_fusion(
            rankings,
            weights=weights,
            k=self.rrf_k,
        )[:top_k]
