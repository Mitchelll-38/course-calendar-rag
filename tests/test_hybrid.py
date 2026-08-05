import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from course_calendar_rag.bm25 import SearchResult  # noqa: E402
from course_calendar_rag.hybrid import HybridIndex, reciprocal_rank_fusion  # noqa: E402


def result(item_id: str) -> SearchResult:
    return SearchResult(item_id, 1.0, {"id": item_id})


class HybridTests(unittest.TestCase):
    def test_rrf_rewards_consensus(self) -> None:
        fused = reciprocal_rank_fusion(
            [[result("lexical"), result("shared")], [result("dense"), result("shared")]], k=60
        )
        self.assertEqual(fused[0].passage_id, "shared")

    def test_weight_changes_ranking(self) -> None:
        fused = reciprocal_rank_fusion(
            [[result("lexical")], [result("dense")]], weights=[2.0, 1.0]
        )
        self.assertEqual(fused[0].passage_id, "lexical")

    def test_hybrid_course_code_lookup(self) -> None:
        passages = [
            {"id": "course", "title": "COMPSCI 2C03", "text": "Data structures."},
            {"id": "other", "title": "Waitlists", "text": "Registration procedure."},
        ]
        self.assertEqual(HybridIndex(passages).search("COMP SCI 2C03")[0].passage_id, "course")


if __name__ == "__main__":
    unittest.main()
