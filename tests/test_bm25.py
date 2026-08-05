import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from course_calendar_rag.bm25 import BM25Index  # noqa: E402
from course_calendar_rag.tokenize import normalize_course_codes, tokenize  # noqa: E402


PASSAGES = [
    {"id": "course", "title": "COMPSCI 2C03", "text": "Data structures and algorithms."},
    {"id": "waitlist", "title": "Waitlists", "text": "A waitlist place is not registration."},
]


class BM25Tests(unittest.TestCase):
    def test_exact_course_code_ranks_first(self) -> None:
        results = BM25Index(PASSAGES).search("COMPSCI 2C03")
        self.assertEqual(results[0].passage_id, "course")

    def test_spaced_department_course_code_ranks_first(self) -> None:
        results = BM25Index(PASSAGES).search("COMP SCI 2C03")
        self.assertEqual(results[0].passage_id, "course")

    def test_course_code_normalization(self) -> None:
        self.assertEqual(normalize_course_codes("Take COMP SCI 2C03 next"), "Take COMPSCI2C03 next")
        self.assertIn("compsci2c03", tokenize("COMP SCI 2C03"))

    def test_invalid_top_k_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            BM25Index(PASSAGES).search("waitlist", top_k=0)


if __name__ == "__main__":
    unittest.main()
