"""Feature-based reranking of candidates from any retrieval index."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Protocol

from .bm25 import SearchResult
from .tokenize import tokenize


class Retriever(Protocol):
    def search(self, query: str, *, top_k: int = 5) -> list[SearchResult]: ...


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class FeatureReranker:
    title_weight: float = 2.0
    body_weight: float = 1.0
    exact_code_bonus: float = 3.0

    def __post_init__(self) -> None:
        values = (self.title_weight, self.body_weight, self.exact_code_bonus)
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("reranker weights must be nonnegative finite numbers")

    def score(self, query: str, passage: dict[str, Any]) -> float:
        query_tokens = set(tokenize(query))
        title_tokens = set(tokenize(passage["title"]))
        body_tokens = set(tokenize(passage["text"]))
        denominator = max(len(query_tokens), 1)
        score = (
            self.title_weight * len(query_tokens & title_tokens) / denominator
            + self.body_weight * len(query_tokens & body_tokens) / denominator
        )
        course_code = passage.get("course_code")
        code_tokens = set(tokenize(course_code)) if course_code else set()
        if code_tokens and code_tokens <= query_tokens:
            score += self.exact_code_bonus
        return score

    def rerank(self, query: str, candidates: list[SearchResult]) -> list[SearchResult]:
        """Score candidates and break equal scores by passage ID."""
        scored: list[SearchResult] = []
        seen: set[str] = set()
        for candidate in candidates:
            if candidate.passage_id in seen:
                continue
            seen.add(candidate.passage_id)
            scored.append(SearchResult(candidate.passage_id, self.score(query, candidate.passage), candidate.passage))
        return sorted(scored, key=lambda result: (-result.score, result.passage_id))


class RerankingIndex:
    def __init__(
        self,
        retriever: Retriever,
        *,
        reranker: FeatureReranker | None = None,
        candidate_k: int = 10,
    ) -> None:
        _positive_integer(candidate_k, "candidate_k")
        self.retriever = retriever
        self.reranker = reranker if reranker is not None else FeatureReranker()
        self.candidate_k = candidate_k

    def search(self, query: str, *, top_k: int = 5) -> list[SearchResult]:
        _positive_integer(top_k, "top_k")
        candidates = self.retriever.search(query, top_k=max(top_k, self.candidate_k))
        return self.reranker.rerank(query, candidates)[:top_k]
