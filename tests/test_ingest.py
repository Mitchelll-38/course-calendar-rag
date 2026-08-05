import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from course_calendar_rag.ingest import assert_allowed_url, parse_calendar_html, write_jsonl  # noqa: E402


class IngestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        html = (ROOT / "tests" / "fixtures" / "calendar_page.html").read_text(encoding="utf-8")
        cls.passages = parse_calendar_html(
            html,
            "https://academiccalendars.romcmaster.ca/content.php?catoid=53&navoid=10776",
            "Wed, 05 Aug 2026 12:00:00 GMT",
        )

    def test_parser_preserves_source_and_sections(self) -> None:
        self.assertEqual(len(self.passages), 2)
        self.assertEqual(self.passages[0].section, "Course registration")
        self.assertTrue(all(item.source.startswith("https://academiccalendars.romcmaster.ca/") for item in self.passages))

    def test_live_passages_are_not_synthetic(self) -> None:
        self.assertTrue(all(item.is_synthetic is False for item in self.passages))

    def test_ids_and_hashes_are_deterministic(self) -> None:
        self.assertTrue(all(item.id.endswith(item.content_sha256[:16]) for item in self.passages))

    def test_non_allowlisted_hosts_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            assert_allowed_url("https://example.com/calendar")

    def test_jsonl_writer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "passages.jsonl"
            write_jsonl(self.passages, output)
            self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 2)


if __name__ == "__main__":
    unittest.main()
