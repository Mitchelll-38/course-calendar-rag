"""Per-question changes between two retrieval evaluations."""

from __future__ import annotations

from typing import Any

from .evaluation import recall_at_k, reciprocal_rank


def _index_details(details: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in details:
        if "question_id" not in row:
            raise ValueError("every evaluation detail needs a question_id")
        question_id = row["question_id"]
        if question_id in indexed:
            raise ValueError(f"duplicate question_id: {question_id}")
        indexed[question_id] = row
    return indexed


def _first_ranks(ranked_ids: list[str]) -> dict[str, int]:
    ranks: dict[str, int] = {}
    for rank, passage_id in enumerate(ranked_ids, start=1):
        ranks.setdefault(passage_id, rank)
    return ranks


def compare_evaluations(
    baseline_details: list[dict[str, Any]],
    candidate_details: list[dict[str, Any]],
    *,
    k: int = 5,
) -> list[dict[str, Any]]:
    """Compare matching questions, preserving the baseline question and gold order.

    Ranks and reciprocal rank consider the full result lists; gain, loss, and
    recall consider only the first ``k`` results. Deltas are candidate minus
    baseline.
    """
    if k < 1:
        raise ValueError("k must be at least 1")

    baseline = _index_details(baseline_details)
    candidate = _index_details(candidate_details)
    if baseline.keys() != candidate.keys():
        raise ValueError("baseline and candidate question_id sets must match")

    comparisons: list[dict[str, Any]] = []
    for question_id, before in baseline.items():
        after = candidate[question_id]
        gold_ids = list(dict.fromkeys(before["gold_passage_ids"]))
        gold_set = set(gold_ids)
        if not gold_set:
            raise ValueError(f"question {question_id} has no gold_passage_ids")
        if gold_set != set(after["gold_passage_ids"]):
            raise ValueError(f"question {question_id} has mismatched gold_passage_ids")

        before_ranked = before["ranked_passage_ids"]
        after_ranked = after["ranked_passage_ids"]
        before_ranks = _first_ranks(before_ranked)
        after_ranks = _first_ranks(after_ranked)
        before_hits = set(before_ranked[:k]) & gold_set
        after_hits = set(after_ranked[:k]) & gold_set
        comparisons.append(
            {
                "question_id": question_id,
                "gained_relevant_passage_ids": [
                    passage_id for passage_id in gold_ids if passage_id in after_hits - before_hits
                ],
                "lost_relevant_passage_ids": [
                    passage_id for passage_id in gold_ids if passage_id in before_hits - after_hits
                ],
                "gold_passage_ranks": [
                    {
                        "passage_id": passage_id,
                        "baseline_rank": before_ranks.get(passage_id),
                        "candidate_rank": after_ranks.get(passage_id),
                    }
                    for passage_id in gold_ids
                ],
                "delta_reciprocal_rank": reciprocal_rank(after_ranked, gold_set)
                - reciprocal_rank(before_ranked, gold_set),
                "delta_recall_at_k": recall_at_k(after_ranked, gold_set, k)
                - recall_at_k(before_ranked, gold_set, k),
            }
        )
    return comparisons
