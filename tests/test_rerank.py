import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from course_calendar_rag.bm25 import SearchResult  # noqa: E402
from course_calendar_rag.rerank import FeatureReranker, RerankingIndex  # noqa: E402


class StubRetriever:
    def __init__(self, results):
        self.results = results

    def search(self, query, *, top_k=5):
        return self.results[:top_k]


class RerankingTests(unittest.TestCase):
    def test_exact_course_code_receives_bonus(self) -> None:
        course = SearchResult("course", 0.1, {"title": "COMPSCI 2C03", "text": "Algorithms", "course_code": "COMPSCI 2C03"})
        generic = SearchResult("generic", 1.0, {"title": "Course registration", "text": "Registration help"})
        ranked = FeatureReranker().rerank("Can I take COMPSCI 2C03?", [generic, course])
        self.assertEqual(ranked[0].passage_id, "course")

    def test_title_overlap_is_preferred(self) -> None:
        title = SearchResult("title", 0.0, {"title": "Course waitlists", "text": "Rules"})
        body = SearchResult("body", 0.0, {"title": "Registration", "text": "Course waitlists"})
        self.assertEqual(FeatureReranker().rerank("course waitlists", [body, title])[0].passage_id, "title")

    def test_wrapper_limits_output(self) -> None:
        items = [SearchResult(str(i), 0.0, {"title": str(i), "text": "text"}) for i in range(4)]
        self.assertEqual(len(RerankingIndex(StubRetriever(items)).search("text", top_k=2)), 2)


if __name__ == "__main__":
    unittest.main()
