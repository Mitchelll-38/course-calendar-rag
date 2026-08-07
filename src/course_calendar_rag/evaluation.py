"""Retrieval metrics and evaluation runner."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean
from time import perf_counter
from typing import Any, Iterable

from .bm25 import BM25Index


@dataclass(frozen=True)
class EvaluationSummary:
    question_count: int
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr: float
    mean_latency_ms: float


def reciprocal_rank(ranked_ids: list[str], gold_ids: set[str]) -> float:
    for rank, passage_id in enumerate(ranked_ids, start=1):
        if passage_id in gold_ids:
            return 1.0 / rank
    return 0.0


def recall_at_k(ranked_ids: list[str], gold_ids: set[str], k: int) -> float:
    if not gold_ids:
        raise ValueError("gold_ids cannot be empty")
    return len(set(ranked_ids[:k]) & gold_ids) / len(gold_ids)


def evaluate(
    index: BM25Index, questions: Iterable[dict[str, Any]], *, top_k: int = 5
) -> tuple[EvaluationSummary, list[dict[str, Any]]]:
    question_list = list(questions)
    if not question_list:
        raise ValueError("at least one question is required")

    details: list[dict[str, Any]] = []
    latencies: list[float] = []
    for item in question_list:
        started = perf_counter()
        results = index.search(item["question"], top_k=top_k)
        latency_ms = (perf_counter() - started) * 1000
        latencies.append(latency_ms)
        ranked_ids = [result.passage_id for result in results]
        gold_ids = set(item["gold_passage_ids"])
        details.append(
            {
                "question_id": item["id"],
                "ranked_passage_ids": ranked_ids,
                "gold_passage_ids": item["gold_passage_ids"],
                "recall_at_1": recall_at_k(ranked_ids, gold_ids, 1),
                "recall_at_3": recall_at_k(ranked_ids, gold_ids, 3),
                "recall_at_5": recall_at_k(ranked_ids, gold_ids, 5),
                "reciprocal_rank": reciprocal_rank(ranked_ids, gold_ids),
            }
        )

    summary = EvaluationSummary(
        question_count=len(question_list),
        recall_at_1=mean(row["recall_at_1"] for row in details),
        recall_at_3=mean(row["recall_at_3"] for row in details),
        recall_at_5=mean(row["recall_at_5"] for row in details),
        mrr=mean(row["reciprocal_rank"] for row in details),
        mean_latency_ms=mean(latencies),
    )
    return summary, details


def summary_as_dict(summary: EvaluationSummary) -> dict[str, Any]:
    result = asdict(summary)
    for key in ("recall_at_1", "recall_at_3", "recall_at_5", "mrr"):
        result[key] = round(result[key], 4)
    result["mean_latency_ms"] = round(result["mean_latency_ms"], 4)
    return result

