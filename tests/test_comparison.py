import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from course_calendar_rag.comparison import compare_evaluations  # noqa: E402


def detail(question_id, gold, ranked):
    return {
        "question_id": question_id,
        "gold_passage_ids": gold,
        "ranked_passage_ids": ranked,
    }


class ComparisonTests(unittest.TestCase):
    def test_multi_gold_gain_loss_and_order_independent_matching(self) -> None:
        baseline = [
            detail("q1", ["g1", "g2", "g3"], ["g1", "x", "g2"]),
            detail("q2", ["a"], ["x", "a"]),
        ]
        candidate = [
            detail("q2", ["a"], ["a", "x"]),
            detail("q1", ["g3", "g2", "g1"], ["g3", "g2", "g1"]),
        ]

        result = compare_evaluations(baseline, candidate, k=2)

        self.assertEqual([row["question_id"] for row in result], ["q1", "q2"])
        self.assertEqual(result[0]["gained_relevant_passage_ids"], ["g2", "g3"])
        self.assertEqual(result[0]["lost_relevant_passage_ids"], ["g1"])
        self.assertEqual(
            result[0]["gold_passage_ranks"],
            [
                {"passage_id": "g1", "baseline_rank": 1, "candidate_rank": 3},
                {"passage_id": "g2", "baseline_rank": 3, "candidate_rank": 2},
                {"passage_id": "g3", "baseline_rank": None, "candidate_rank": 1},
            ],
        )
        self.assertAlmostEqual(result[0]["delta_recall_at_k"], 1 / 3)
        self.assertEqual(result[0]["delta_reciprocal_rank"], 0)
        self.assertEqual(result[1]["delta_reciprocal_rank"], 0.5)
        self.assertEqual(result[1]["delta_recall_at_k"], 0)
        json.dumps(result)

    def test_no_change(self) -> None:
        row = detail("q", ["a"], ["x", "a"])
        result = compare_evaluations([row], [row], k=1)
        self.assertEqual(result[0]["gained_relevant_passage_ids"], [])
        self.assertEqual(result[0]["lost_relevant_passage_ids"], [])
        self.assertEqual(result[0]["delta_reciprocal_rank"], 0)
        self.assertEqual(result[0]["delta_recall_at_k"], 0)
        self.assertEqual(result[0]["gold_passage_ranks"][0]["baseline_rank"], 2)

    def test_rejects_missing_or_duplicate_question_ids(self) -> None:
        row = detail("q", ["a"], ["a"])
        other = detail("other", ["a"], ["a"])
        with self.assertRaisesRegex(ValueError, "sets must match"):
            compare_evaluations([row], [other])
        with self.assertRaisesRegex(ValueError, "duplicate question_id"):
            compare_evaluations([row, row], [row])
        with self.assertRaisesRegex(ValueError, "duplicate question_id"):
            compare_evaluations([row], [row, row])
        with self.assertRaisesRegex(ValueError, "needs a question_id"):
            compare_evaluations([{"gold_passage_ids": ["a"]}], [row])

    def test_rejects_mismatched_gold_and_invalid_k(self) -> None:
        row = detail("q", ["a"], ["a"])
        with self.assertRaisesRegex(ValueError, "mismatched gold"):
            compare_evaluations([row], [detail("q", ["b"], ["b"])])
        with self.assertRaisesRegex(ValueError, "at least 1"):
            compare_evaluations([row], [row], k=0)


if __name__ == "__main__":
    unittest.main()
