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


class CompareCliTests(unittest.TestCase):
    def run_cli(self, *args: str) -> tuple[int, str]:
        output = io.StringIO()
        with patch.object(sys, "argv", ["course-calendar-rag", *args]):
            with contextlib.redirect_stdout(output):
                exit_code = main()
        return exit_code, output.getvalue()

    def test_compare_reports_q015_regression_and_aggregate_deltas(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "nested" / "comparison.json"
            exit_code, output = self.run_cli("compare", "--output", str(destination))
            self.assertEqual(exit_code, 0)
            self.assertIn(str(destination), output)
            payload = json.loads(destination.read_text(encoding="utf-8"))

        self.assertEqual(payload["baseline"]["retriever"], "bm25")
        self.assertEqual(payload["hybrid"]["retriever"], "hybrid")
        self.assertEqual(payload["baseline"]["summary"]["question_count"], 20)
        self.assertEqual(payload["hybrid"]["summary"]["question_count"], 20)
        self.assertEqual(payload["aggregate_deltas"]["recall_at_5"], -0.05)
        self.assertEqual(
            payload["aggregate_deltas"]["mrr"],
            round(payload["hybrid"]["summary"]["mrr"] - payload["baseline"]["summary"]["mrr"], 4),
        )
        self.assertEqual(len(payload["questions"]), 20)
        q015 = next(row for row in payload["questions"] if row["question_id"] == "q015")
        self.assertEqual(q015["lost_relevant_passage_ids"], ["reg-reserved-001"])
        self.assertLess(q015["delta_recall_at_k"], 0)

    def test_compare_requires_depth_of_at_least_five(self) -> None:
        for depth in (0, 1, 4):
            with self.subTest(depth=depth), patch.object(sys, "argv", ["course-calendar-rag", "compare", "--top-k", str(depth)]):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as raised:
                        main()
            self.assertEqual(raised.exception.code, 2)

    def test_extra_rank_depth_preserves_recall_at_five(self) -> None:
        exit_code, output = self.run_cli("compare", "--top-k", "10")
        self.assertEqual(exit_code, 0)
        payload = json.loads(output)
        self.assertEqual(payload["aggregate_deltas"]["recall_at_5"], -0.05)
        q015 = next(row for row in payload["questions"] if row["question_id"] == "q015")
        self.assertEqual(q015["lost_relevant_passage_ids"], ["reg-reserved-001"])
        self.assertEqual(q015["gold_passage_ranks"][0]["baseline_rank"], 2)
        self.assertGreater(q015["gold_passage_ranks"][0]["candidate_rank"], 5)

    def test_compare_does_not_overwrite_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "existing.json"
            destination.write_text("original", encoding="utf-8")
            with patch.object(sys, "argv", ["course-calendar-rag", "compare", "--output", str(destination)]):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as raised:
                        main()
            self.assertEqual(raised.exception.code, 2)
            self.assertEqual(destination.read_text(encoding="utf-8"), "original")

    def test_existing_command_defaults_still_use_bm25(self) -> None:
        parser = build_parser()
        self.assertEqual(parser.parse_args(["search", "reserved seats"]).retriever, "bm25")
        self.assertEqual(parser.parse_args(["evaluate"]).retriever, "bm25")
        self.assertEqual(parser.parse_args(["compare"]).top_k, 5)


if __name__ == "__main__":
    unittest.main()
