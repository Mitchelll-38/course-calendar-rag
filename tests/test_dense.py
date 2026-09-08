import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from course_calendar_rag.dense import DenseIndex, HashingEmbedder, cosine_similarity  # noqa: E402


class DenseTests(unittest.TestCase):
    def test_vectors_are_normalized_and_deterministic(self) -> None:
        embedder = HashingEmbedder(dimensions=64)
        first, second = embedder.encode(["course waitlist", "course waitlist"])
        self.assertEqual(first, second)
        self.assertAlmostEqual(cosine_similarity(first, first), 1.0)

    def test_related_lexical_features_rank_first(self) -> None:
        passages = [
            {"id": "waitlist", "title": "Waitlists", "text": "A waitlist is not registration."},
            {"id": "repeat", "title": "Repeats", "text": "Both attempts remain on the record."},
        ]
        result = DenseIndex(passages).search("Am I registered on a waitlist?")[0]
        self.assertEqual(result.passage_id, "waitlist")

    def test_empty_index_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            DenseIndex([])


if __name__ == "__main__":
    unittest.main()
