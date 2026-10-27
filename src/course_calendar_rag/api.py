"""Local API backed by the synthetic course calendar fixture."""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .bm25 import BM25Index, SearchResult
from .cli import DEFAULT_PASSAGES, DEFAULT_QUESTIONS, load_jsonl
from .dense import DenseIndex
from .evaluation import evaluate, summary_as_dict
from .hybrid import HybridIndex
from .rerank import RerankingIndex


RetrieverName = Literal["bm25", "dense", "hybrid", "reranked"]
DISCLAIMER = "Synthetic sample data only. Verify academic requirements with an official calendar."


class AnswerRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    retriever: RetrieverName = "hybrid"
    top_k: int = Field(default=5, ge=1, le=10)


@lru_cache(maxsize=1)
def _indexes() -> dict[str, BM25Index | DenseIndex | HybridIndex | RerankingIndex]:
    passages = load_jsonl(DEFAULT_PASSAGES)
    hybrid = HybridIndex(passages)
    return {
        "bm25": BM25Index(passages),
        "dense": DenseIndex(passages),
        "hybrid": hybrid,
        "reranked": RerankingIndex(hybrid),
    }


@lru_cache(maxsize=1)
def _questions() -> list[dict]:
    return load_jsonl(DEFAULT_QUESTIONS)


def _source(result: SearchResult) -> dict:
    passage = result.passage
    return {
        "passage_id": result.passage_id,
        "title": passage["title"],
        "section": passage["section"],
        "source": passage["source"],
        "score": result.score,
        "excerpt": passage["text"],
    }


app = FastAPI(title="Course Calendar RAG Local Demo")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "dataset": "synthetic"}


@app.get("/search")
def search(
    q: Annotated[str, Query(min_length=3, max_length=500)],
    retriever: RetrieverName = "hybrid",
    top_k: Annotated[int, Query(ge=1, le=10)] = 5,
) -> dict:
    query = q.strip()
    # Query validates raw input length; reject whitespace-only input as well.
    if len(query) < 3:
        raise HTTPException(status_code=422, detail="q must contain at least 3 characters")
    results = _indexes()[retriever].search(query, top_k=top_k)
    return {"query": query, "retriever": retriever, "sources": [_source(item) for item in results]}


@app.post("/answer")
def answer(request: AnswerRequest) -> dict:
    question = request.question.strip()
    if len(question) < 3:
        raise HTTPException(status_code=422, detail="question must contain at least 3 characters")
    results = _indexes()[request.retriever].search(question, top_k=request.top_k)
    sources = [_source(item) for item in results]
    return {
        "question": question,
        "retriever": request.retriever,
        "answer": sources[0]["excerpt"] if sources else "",
        "sources": sources,
        "disclaimer": DISCLAIMER,
    }


@app.get("/evaluation")
def evaluation(retriever: RetrieverName = "hybrid") -> dict:
    return _evaluation_result(retriever)


@lru_cache(maxsize=4)
def _evaluation_result(retriever: RetrieverName) -> dict:
    summary, _ = evaluate(_indexes()[retriever], _questions())
    return {"retriever": retriever, "summary": summary_as_dict(summary)}
