import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from course_calendar_rag.bm25 import SearchResult  # noqa: E402
from course_calendar_rag.hybrid import HybridIndex, reciprocal_rank_fusion  # noqa: E402


PASSAGES = [
    {"id": "course", "title": "COMPSCI 2C03", "text": "Data structures and algorithms."},
    {"id": "waitlist", "title": "Waitlists", "text": "A waitlist place is not registration."},
    {"id": "fees", "title": "Fees", "text": "Pay tuition before the deadline."},
]


def result(passage_id: str) -> SearchResult:
    return SearchResult(passage_id, 1.0, {"id": passage_id})


class RRFTests(unittest.TestCase):
    def test_consensus_beats_single_first_place(self) -> None:
        rankings = [
            [result("a"), result("shared")],
            [result("b"), result("shared")],
        ]
        fused = reciprocal_rank_fusion(rankings)
        self.assertEqual([item.passage_id for item in fused], ["shared", "a", "b"])
        self.assertAlmostEqual(fused[0].score, 2 / 62)
        self.assertEqual(fused[0].passage, {"id": "shared"})

    def test_weights_change_ranking_and_zero_branch_is_excluded(self) -> None:
        rankings = [[result("lexical"), result("dense")], [result("dense"), result("lexical")]]
        self.assertEqual(reciprocal_rank_fusion(rankings, [2, 1])[0].passage_id, "lexical")
        self.assertEqual(reciprocal_rank_fusion(rankings, [1, 2])[0].passage_id, "dense")
        self.assertEqual([r.passage_id for r in reciprocal_rank_fusion([[result("a")], [result("b")]], [1, 0])], ["a"])

    def test_duplicate_ids_only_score_once_per_ranking(self) -> None:
        fused = reciprocal_rank_fusion([[result("a"), result("a")], [result("a")]], k=1)
        self.assertEqual(len(fused), 1)
        self.assertAlmostEqual(fused[0].score, 1.0)
        distinct = reciprocal_rank_fusion([[result("a"), result("a"), result("b")]], k=1)
        self.assertAlmostEqual(distinct[1].score, 1 / 3)

    def test_equal_scores_break_ties_by_id(self) -> None:
        rankings = [[result("b"), result("a")], [result("a"), result("b")]]
        self.assertEqual([item.passage_id for item in reciprocal_rank_fusion(rankings)], ["a", "b"])

    def test_invalid_parameters_are_rejected(self) -> None:
        rankings = [[result("a")], [result("b")]]
        for k in (0, -1, 1.5, math.inf):
            with self.subTest(k=k), self.assertRaises(ValueError):
                reciprocal_rank_fusion(rankings, k=k)
        for weights in ([1], [-1, 1], [math.inf, 1], [math.nan, 1], [0, 0], [True, 1]):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                reciprocal_rank_fusion(rankings, weights)
        self.assertEqual(reciprocal_rank_fusion([]), [])


class HybridTests(unittest.TestCase):
    def test_zero_dense_weight_does_not_call_embedder(self) -> None:
        class FailingEmbedder:
            dimensions = 1

            def encode(self, texts: list[str]) -> list[list[float]]:
                raise AssertionError("unused dense embedder was called")

        index = HybridIndex(PASSAGES, embedder=FailingEmbedder(), dense_weight=0)
        self.assertEqual(index.search("COMPSCI 2C03")[0].passage_id, "course")

    def test_course_code_lookup(self) -> None:
        index = HybridIndex(PASSAGES)
        self.assertEqual(index.search("COMP SCI 2C03")[0].passage_id, "course")

    def test_requested_top_k_exceeds_candidate_depth(self) -> None:
        index = HybridIndex(PASSAGES, candidate_k=1)
        self.assertEqual(len(index.search("course", top_k=3)), 3)

    def test_invalid_parameters_are_rejected(self) -> None:
        for kwargs in ({"candidate_k": 0}, {"rrf_k": 0}, {"lexical_weight": -1},
                       {"dense_weight": math.nan}, {"lexical_weight": 0, "dense_weight": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                HybridIndex(PASSAGES, **kwargs)
        with self.assertRaises(ValueError):
            HybridIndex(PASSAGES).search("course", top_k=0)
        with self.assertRaises(ValueError):
            HybridIndex([])
        with self.assertRaises(ValueError):
            HybridIndex([PASSAGES[0], PASSAGES[0]])


if __name__ == "__main__":
    unittest.main()
