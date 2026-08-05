"""Candidate reranking with interpretable query-passage features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .bm25 import SearchResult
from .tokenize import tokenize


class Retriever(Protocol):
    def search(self, query: str, *, top_k: int = 5) -> list[SearchResult]: ...


@dataclass(frozen=True)
class FeatureReranker:
    title_weight: float = 2.0
    body_weight: float = 1.0
    exact_code_bonus: float = 3.0

    def score(self, query: str, result: SearchResult) -> float:
        query_terms = set(tokenize(query))
        title_terms = set(tokenize(result.passage["title"]))
        body_terms = set(tokenize(result.passage["text"]))
        title_overlap = len(query_terms & title_terms) / max(len(query_terms), 1)
        body_overlap = len(query_terms & body_terms) / max(len(query_terms), 1)
        course_code = result.passage.get("course_code")
        code_bonus = 0.0
        if course_code and set(tokenize(course_code)) <= query_terms:
            code_bonus = self.exact_code_bonus
        return self.title_weight * title_overlap + self.body_weight * body_overlap + code_bonus

    def rerank(self, query: str, candidates: list[SearchResult]) -> list[SearchResult]:
        reranked = [
            SearchResult(item.passage_id, self.score(query, item), item.passage)
            for item in candidates
        ]
        reranked.sort(key=lambda result: (-result.score, result.passage_id))
        return reranked


class RerankingIndex:
    def __init__(
        self, retriever: Retriever, *, reranker: FeatureReranker | None = None, candidate_k: int = 10
    ) -> None:
        self.retriever = retriever
        self.reranker = reranker or FeatureReranker()
        self.candidate_k = candidate_k

    def search(self, query: str, *, top_k: int = 5) -> list[SearchResult]:
        candidates = self.retriever.search(query, top_k=max(top_k, self.candidate_k))
        return self.reranker.rerank(query, candidates)[:top_k]
