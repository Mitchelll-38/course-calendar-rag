import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from course_calendar_rag.bm25 import BM25Index  # noqa: E402
from course_calendar_rag.evaluation import evaluate, recall_at_k, reciprocal_rank  # noqa: E402


class EvaluationTests(unittest.TestCase):
    def test_reciprocal_rank(self) -> None:
        self.assertEqual(reciprocal_rank(["a", "gold", "b"], {"gold"}), 0.5)
        self.assertEqual(reciprocal_rank(["a", "b"], {"gold"}), 0.0)

    def test_recall_at_k_supports_multiple_gold_passages(self) -> None:
        self.assertEqual(recall_at_k(["a", "b", "c"], {"a", "c"}, 2), 0.5)
        self.assertEqual(recall_at_k(["a", "b", "c"], {"a", "c"}, 3), 1.0)

    def test_evaluate_returns_summary_and_details(self) -> None:
        passages = [{"id": "p1", "title": "Waitlist", "text": "A waitlist is not registration."}]
        questions = [{"id": "q1", "question": "Am I registered on a waitlist?", "gold_passage_ids": ["p1"]}]
        summary, details = evaluate(BM25Index(passages), questions)
        self.assertEqual(summary.question_count, 1)
        self.assertEqual(summary.mrr, 1.0)
        self.assertEqual(details[0]["ranked_passage_ids"], ["p1"])


if __name__ == "__main__":
    unittest.main()
