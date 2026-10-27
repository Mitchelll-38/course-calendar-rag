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


class LocalPassageCliTests(unittest.TestCase):
    def run_cli(self, *args: str) -> tuple[int, str, str]:
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.object(sys, "argv", ["calendar-rag", *args]):
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                try:
                    code = main()
                except SystemExit as exc:
                    code = exc.code
        return code, stdout.getvalue(), stderr.getvalue()

    def write(self, directory: str, rows: list[object]) -> Path:
        path = Path(directory) / "passages.jsonl"
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        return path

    def test_all_retrievers_search_local_export(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            path = self.write(directory, [
                {"id": "local-1", "title": "Registration", "text": "Students register in October."},
                {"id": "local-2", "title": "Graduation", "text": "Apply to graduate in May."},
            ])
            for retriever in ("bm25", "dense", "hybrid", "reranked"):
                with self.subTest(retriever=retriever):
                    code, output, error = self.run_cli(
                        "search", "graduate May", "--passages", str(path),
                        "--retriever", retriever, "--top-k", "1",
                    )
                    self.assertEqual(code, 0, error)
                    self.assertIn("local-2", output)

    def test_bad_records_report_file_and_line(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            cases = [([{"title": "Missing ID", "text": "Body"}], "id must be a nonempty string"),
                     ([{"id": "x", "title": "Title", "text": 3}], "text must be a nonempty string"),
                     ([{"id": "x", "title": "Title", "text": "Body"},
                       {"id": "x", "title": "Other", "text": "Other"}], "duplicate passage id")]
            for rows, message in cases:
                with self.subTest(message=message):
                    path = self.write(directory, rows)
                    code, _, error = self.run_cli("search", "query", "--passages", str(path))
                    self.assertEqual(code, 2)
                    self.assertIn(str(path), error)
                    self.assertIn(message, error)

    def test_invalid_json_empty_file_and_missing_file(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            path = Path(directory) / "passages.jsonl"
            for content, message in (("{oops\n", "invalid JSON"), ("\n", "no passages found")):
                path.write_text(content, encoding="utf-8")
                code, _, error = self.run_cli("search", "query", "--passages", str(path))
                self.assertEqual(code, 2)
                self.assertIn(message, error)
            path.unlink()
            code, _, error = self.run_cli("search", "query", "--passages", str(path))
            self.assertEqual(code, 2)
            self.assertIn("cannot load passages", error)

    def test_punctuation_only_record_is_rejected_before_indexing(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            path = self.write(directory, [{"id": "x", "title": "---", "text": "!!!"}])
            for retriever in ("bm25", "dense", "hybrid", "reranked"):
                with self.subTest(retriever=retriever):
                    code, _, error = self.run_cli(
                        "search", "registration", "--passages", str(path), "--retriever", retriever,
                    )
                    self.assertEqual(code, 2)
                    self.assertIn(f"{path}:1: passage has no searchable tokens", error)

    def test_evaluation_does_not_accept_external_passages(self) -> None:
        code, _, error = self.run_cli("evaluate", "--passages", "anything.jsonl")
        self.assertEqual(code, 2)
        self.assertIn("unrecognized arguments", error)


if __name__ == "__main__":
    unittest.main()
