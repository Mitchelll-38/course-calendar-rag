import math
import sys
import unittest
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from course_calendar_rag.dense import DenseIndex, HashingEmbedder  # noqa: E402


PASSAGES = [
    {"id": "course", "title": "COMPSCI 2C03", "text": "Data structures and algorithms."},
    {"id": "waitlist", "title": "Waitlists", "text": "A waitlist place is not registration."},
]


class StubEmbedder:
    dimensions = 2

    def __init__(self, passage_vectors: list, query_vectors: list | None = None) -> None:
        self.passage_vectors = passage_vectors
        self.query_vectors = query_vectors if query_vectors is not None else [[1.0, 0.0]]
        self.calls = 0

    def encode(self, texts: list[str]) -> list:
        self.calls += 1
        return self.passage_vectors if self.calls == 1 else self.query_vectors


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

    def test_valid_custom_embedder_scores_every_passage(self) -> None:
        embedder = StubEmbedder([[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0]])
        results = DenseIndex(PASSAGES, embedder=embedder).search("waitlist")
        self.assertEqual([result.passage_id for result in results], ["waitlist", "course"])

    def test_decimal_embeddings_are_normalized_before_scoring(self) -> None:
        embedder = StubEmbedder(
            [[Decimal("1"), Decimal("0")], [Decimal("0"), Decimal("1")]],
            [[Decimal("0"), Decimal("1")]],
        )
        results = DenseIndex(PASSAGES, embedder=embedder).search("waitlist")
        self.assertEqual([result.score for result in results], [1.0, 0.0])

    def test_nonfinite_similarity_is_rejected(self) -> None:
        embedder = StubEmbedder([[1e308, 0.0], [0.0, 1.0]], [[1e308, 0.0]])
        with self.assertRaisesRegex(ValueError, "similarity score must be finite"):
            DenseIndex(PASSAGES, embedder=embedder).search("waitlist")

    def test_empty_index_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            DenseIndex([])

    def test_duplicate_passage_ids_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            DenseIndex([PASSAGES[0], PASSAGES[0]])

    def test_invalid_passage_embeddings_are_rejected(self) -> None:
        cases = [
            ([[1.0, 0.0]], "expected 2"),
            ([[1.0, 0.0]] * 3, "expected 2"),
            ([[1.0], [0.0, 1.0]], "expected 2"),
            ([[1.0, 0.0], [float("nan"), 1.0]], "finite numbers"),
            ([[1.0, 0.0], [float("inf"), 1.0]], "finite numbers"),
            ([[1.0, 0.0], [True, 1.0]], "finite numbers"),
        ]
        for vectors, message in cases:
            with self.subTest(vectors=vectors), self.assertRaisesRegex(ValueError, message):
                DenseIndex(PASSAGES, embedder=StubEmbedder(vectors))

    def test_invalid_query_embeddings_are_rejected(self) -> None:
        cases = [
            ([], "expected 1"),
            ([[1.0, 0.0]] * 2, "expected 1"),
            ([[1.0]], "expected 2"),
            ([[float("nan"), 0.0]], "finite numbers"),
        ]
        for vectors, message in cases:
            with self.subTest(vectors=vectors), self.assertRaisesRegex(ValueError, message):
                DenseIndex(PASSAGES, embedder=StubEmbedder([[1.0, 0.0]] * 2, vectors)).search(
                    "waitlist"
                )

    def test_invalid_embedder_dimensions_are_rejected(self) -> None:
        for dimensions in (0, -1, True, 2.0):
            with self.subTest(dimensions=dimensions):
                embedder = StubEmbedder([[1.0, 0.0]] * 2)
                embedder.dimensions = dimensions
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    DenseIndex(PASSAGES, embedder=embedder)


if __name__ == "__main__":
    unittest.main()
