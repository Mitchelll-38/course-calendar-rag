"""HTTP API for retrieval, grounded answers, and evaluation."""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated, Literal

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .bm25 import BM25Index
from .cli import DEFAULT_PASSAGES, DEFAULT_QUESTIONS, load_jsonl
from .dense import DenseIndex
from .evaluation import evaluate, summary_as_dict
from .hybrid import HybridIndex
from .rerank import RerankingIndex


RetrieverName = Literal["bm25", "dense", "hybrid", "reranked"]


class Source(BaseModel):
    passage_id: str
    title: str
    section: str
    source: str
    score: float
    excerpt: str


class SearchResponse(BaseModel):
    query: str
    retriever: RetrieverName
    sources: list[Source]


class AnswerRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    retriever: RetrieverName = "hybrid"
    top_k: int = Field(default=5, ge=1, le=10)


class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source]
    disclaimer: str


@lru_cache(maxsize=1)
def passages() -> list[dict]:
    return load_jsonl(DEFAULT_PASSAGES)


@lru_cache(maxsize=4)
def get_retriever(name: RetrieverName):
    corpus = passages()
    if name == "bm25":
        return BM25Index(corpus)
    if name == "dense":
        return DenseIndex(corpus)
    hybrid = HybridIndex(corpus)
    return RerankingIndex(hybrid) if name == "reranked" else hybrid


def serialize_sources(results) -> list[Source]:
    return [
        Source(
            passage_id=item.passage_id,
            title=item.passage["title"],
            section=item.passage["section"],
            source=item.passage["source"],
            score=round(item.score, 6),
            excerpt=item.passage["text"],
        )
        for item in results
    ]


app = FastAPI(
    title="Course Calendar RAG API",
    description="Grounded retrieval over the synthetic academic-calendar fixture.",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "dataset": "synthetic"}


@app.get("/search", response_model=SearchResponse)
def search(
    q: Annotated[str, Query(min_length=2, max_length=500)],
    retriever: RetrieverName = "hybrid",
    top_k: Annotated[int, Query(ge=1, le=10)] = 5,
) -> SearchResponse:
    results = get_retriever(retriever).search(q, top_k=top_k)
    return SearchResponse(query=q, retriever=retriever, sources=serialize_sources(results))


@app.post("/answer", response_model=AnswerResponse)
def answer(request: AnswerRequest) -> AnswerResponse:
    results = get_retriever(request.retriever).search(request.question, top_k=request.top_k)
    sources = serialize_sources(results)
    grounded_answer = sources[0].excerpt if sources else "No relevant passage was found."
    return AnswerResponse(
        question=request.question,
        answer=grounded_answer,
        sources=sources,
        disclaimer="Synthetic demo content only; verify academic decisions against the official calendar.",
    )


@app.get("/evaluation")
def evaluation(retriever: RetrieverName = "hybrid") -> dict:
    summary, _ = evaluate(get_retriever(retriever), load_jsonl(DEFAULT_QUESTIONS))
    return {"retriever": retriever, "summary": summary_as_dict(summary)}
