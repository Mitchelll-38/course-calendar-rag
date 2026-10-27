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

from course_calendar_rag.cli import main  # noqa: E402


class TopKCliTests(unittest.TestCase):
    def run_cli(self, *args: str) -> tuple[int, str, str]:
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.object(sys, "argv", ["calendar-rag", *args]):
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                try:
                    code = main()
                except SystemExit as exc:
                    code = exc.code
        return code, stdout.getvalue(), stderr.getvalue()

    def test_search_rejects_nonpositive_depth(self) -> None:
        for depth in ("0", "-1"):
            with self.subTest(depth=depth):
                code, output, error = self.run_cli("search", "waitlist", "--top-k", depth)
                self.assertEqual(code, 2)
                self.assertEqual(output, "")
                self.assertIn("search --top-k must be at least 1", error)
                self.assertNotIn("Traceback", error)

    def test_evaluate_requires_full_recall_at_five_depth_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "results.json"
            for depth in ("0", "-1", "1", "4"):
                with self.subTest(depth=depth):
                    code, output, error = self.run_cli(
                        "evaluate", "--top-k", depth, "--output", str(destination)
                    )
                    self.assertEqual(code, 2)
                    self.assertEqual(output, "")
                    self.assertIn("evaluate --top-k must be at least 5 to measure recall@5", error)
                    self.assertFalse(destination.exists())

    def test_valid_search_and_evaluation_depths(self) -> None:
        code, output, error = self.run_cli("search", "waitlist", "--top-k", "1")
        self.assertEqual(code, 0, error)
        self.assertEqual(len(output.splitlines()), 1)

        results = []
        for depth in ("5", "10"):
            code, output, error = self.run_cli("evaluate", "--top-k", depth)
            self.assertEqual(code, 0, error)
            results.append(json.loads(output))
        self.assertEqual(results[0]["summary"]["recall_at_5"], 0.925)
        self.assertEqual(results[1]["summary"]["recall_at_5"], 0.925)


if __name__ == "__main__":
    unittest.main()
