import sys
import math
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from course_calendar_rag.bm25 import SearchResult  # noqa: E402
from course_calendar_rag.rerank import FeatureReranker, RerankingIndex  # noqa: E402


def result(passage_id: str, title: str, text: str = "", course_code: str = "") -> SearchResult:
    passage = {"id": passage_id, "title": title, "text": text, "course_code": course_code}
    return SearchResult(passage_id, 99.0, passage)


class FeatureRerankerTests(unittest.TestCase):
    def test_invalid_feature_weights_are_rejected(self) -> None:
        for value in (-1, math.inf, math.nan, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                FeatureReranker(title_weight=value)

    def test_empty_normalized_course_code_gets_no_bonus(self) -> None:
        passage = result("course", "Other", course_code="---")
        self.assertEqual(FeatureReranker().score("registration", passage.passage), 0.0)

    def test_duplicate_candidates_are_scored_once(self) -> None:
        candidates = [result("a", "Match"), result("a", "Match"), result("b", "Match")]
        self.assertEqual([item.passage_id for item in FeatureReranker().rerank("match", candidates)], ["a", "b"])

    def test_exact_course_code_bonus_accepts_normalized_spelling(self) -> None:
        reranker = FeatureReranker()
        course = result("course", "Algorithms", course_code="COMPSCI 2C03")
        self.assertEqual(reranker.score("COMP SCI 2C03", course.passage), 3.0)
        self.assertEqual(reranker.score("COMPSCI 2C03", course.passage), 3.0)
        self.assertEqual(reranker.score("COMPSCI 2C04", course.passage), 0.0)

    def test_title_overlap_outweighs_body_overlap(self) -> None:
        candidates = [result("body", "Other", "registration"), result("title", "Registration")]
        ranked = FeatureReranker().rerank("registration", candidates)
        self.assertEqual([item.passage_id for item in ranked], ["title", "body"])
        self.assertEqual([item.score for item in ranked], [2.0, 1.0])

    def test_overlap_uses_fraction_of_distinct_query_tokens(self) -> None:
        passage = result("a", "registration deadline", "registration")
        self.assertEqual(FeatureReranker().score("registration fees", passage.passage), 1.5)
        self.assertEqual(FeatureReranker().score("", passage.passage), 0.0)

    def test_equal_scores_break_ties_by_id(self) -> None:
        ranked = FeatureReranker().rerank("missing", [result("b", ""), result("a", "")])
        self.assertEqual([item.passage_id for item in ranked], ["a", "b"])


class RerankingIndexTests(unittest.TestCase):
    def test_requests_enough_candidates_and_limits_output(self) -> None:
        class StubRetriever:
            def __init__(self) -> None:
                self.depths: list[int] = []

            def search(self, query: str, *, top_k: int = 5) -> list[SearchResult]:
                self.depths.append(top_k)
                return [result(str(i), "match" if i == top_k - 1 else "other") for i in range(top_k)]

        retriever = StubRetriever()
        index = RerankingIndex(retriever, candidate_k=3)
        self.assertEqual([item.passage_id for item in index.search("match", top_k=1)], ["2"])
        self.assertEqual(len(index.search("match", top_k=4)), 4)
        self.assertEqual(retriever.depths, [3, 4])

    def test_invalid_depths_are_rejected(self) -> None:
        for value in (0, -1, 1.5, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                RerankingIndex(object(), candidate_k=value)
        with self.assertRaises(ValueError):
            RerankingIndex(object()).search("query", top_k=0)


if __name__ == "__main__":
    unittest.main()
