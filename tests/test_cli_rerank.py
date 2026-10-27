import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from course_calendar_rag.cli import build_parser, main  # noqa: E402


class RerankCliTests(unittest.TestCase):
    def run_cli(self, *args: str) -> tuple[int, str]:
        output = io.StringIO()
        with patch.object(sys, "argv", ["course-calendar-rag", *args]):
            with contextlib.redirect_stdout(output):
                exit_code = main()
        return exit_code, output.getvalue()

    def test_search_uses_reranked_results(self) -> None:
        exit_code, output = self.run_cli("search", "COMP SCI 2C03", "--retriever", "reranked", "--top-k", "2")
        self.assertEqual(exit_code, 0)
        lines = output.splitlines()
        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[0].startswith("1. course-compsci-2c03 (6.0000)"))
        self.assertTrue(lines[1].startswith("2. "))

    def test_evaluate_writes_reranked_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "nested" / "reranked.json"
            exit_code, output = self.run_cli("evaluate", "--retriever", "reranked", "--output", str(destination))
            self.assertEqual(exit_code, 0)
            self.assertIn(str(destination), output)
            payload = json.loads(destination.read_text(encoding="utf-8"))

        self.assertEqual(payload["retriever"], "reranked")
        self.assertEqual(payload["summary"]["question_count"], 20)
        self.assertEqual(payload["summary"]["recall_at_5"], 0.975)
        self.assertEqual(len(payload["questions"]), 20)
        q015 = next(row for row in payload["questions"] if row["question_id"] == "q015")
        self.assertIn("reg-reserved-001", q015["ranked_passage_ids"][:5])

    def test_search_and_evaluate_still_default_to_bm25(self) -> None:
        parser = build_parser()
        self.assertEqual(parser.parse_args(["search", "reserved seats"]).retriever, "bm25")
        self.assertEqual(parser.parse_args(["evaluate"]).retriever, "bm25")


if __name__ == "__main__":
    unittest.main()
