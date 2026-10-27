import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from course_calendar_rag.ingest import (  # noqa: E402
    assert_allowed_url,
    fetch_and_parse,
    parse_calendar_html,
    write_jsonl,
)


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
        for url in (
            "https://example.com/calendar",
            "http://academiccalendars.romcmaster.ca/calendar",
            "https://user:secret@academiccalendars.romcmaster.ca/calendar",
            "https://academiccalendars.romcmaster.ca:8443/calendar",
        ):
            with self.subTest(url=url), self.assertRaises(ValueError):
                assert_allowed_url(url)

    def test_robots_redirect_is_rejected_without_following(self) -> None:
        requests = []

        def handler(request):
            requests.append(str(request.url))
            return httpx.Response(302, headers={"location": "https://example.com/robots.txt"})

        client = httpx.Client(transport=httpx.MockTransport(handler))
        with patch("course_calendar_rag.ingest.httpx.Client", return_value=client) as make_client:
            with self.assertRaises(PermissionError):
                fetch_and_parse("https://academiccalendars.romcmaster.ca/calendar", delay_seconds=0)
        self.assertEqual(requests, ["https://academiccalendars.romcmaster.ca/robots.txt"])
        self.assertFalse(make_client.call_args.kwargs["follow_redirects"])

    def test_content_redirect_is_rejected_without_following(self) -> None:
        requests = []

        def handler(request):
            requests.append(str(request.url))
            if request.url.path == "/robots.txt":
                return httpx.Response(200, text="User-agent: *\nAllow: /")
            return httpx.Response(302, headers={"location": "https://example.com/calendar"})

        client = httpx.Client(transport=httpx.MockTransport(handler))
        with patch("course_calendar_rag.ingest.httpx.Client", return_value=client), patch(
            "course_calendar_rag.ingest.time.sleep"
        ):
            with self.assertRaises(httpx.HTTPStatusError):
                fetch_and_parse("https://academiccalendars.romcmaster.ca/calendar", delay_seconds=0)
        self.assertEqual(
            requests,
            [
                "https://academiccalendars.romcmaster.ca/robots.txt",
                "https://academiccalendars.romcmaster.ca/calendar",
            ],
        )

    def test_jsonl_writer(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            output = Path(directory) / "passages.jsonl"
            write_jsonl(self.passages, output)
            self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 2)


if __name__ == "__main__":
    unittest.main()
