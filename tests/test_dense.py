import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from course_calendar_rag.dense import DenseIndex, HashingEmbedder  # noqa: E402


PASSAGES = [
    {"id": "course", "title": "COMPSCI 2C03", "text": "Data structures and algorithms."},
    {"id": "waitlist", "title": "Waitlists", "text": "A waitlist place is not registration."},
]


class DenseTests(unittest.TestCase):
    def test_hashing_embedder_requires_positive_dimensions(self) -> None:
        with self.assertRaises(ValueError):
            HashingEmbedder(dimensions=0)

    def test_hashing_embedder_is_deterministic_and_normalized(self) -> None:
        embedder = HashingEmbedder(dimensions=64)
        first, second = embedder.encode(["course waitlist", "course waitlist"])
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)
        self.assertAlmostEqual(math.sqrt(sum(value * value for value in first)), 1.0)

    def test_waitlist_query_ranks_waitlist_first(self) -> None:
        results = DenseIndex(PASSAGES).search("waitlist")
        self.assertEqual(results[0].passage_id, "waitlist")

    def test_empty_index_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            DenseIndex([])

    def test_duplicate_passage_ids_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            DenseIndex([PASSAGES[0], PASSAGES[0]])


if __name__ == "__main__":
    unittest.main()
