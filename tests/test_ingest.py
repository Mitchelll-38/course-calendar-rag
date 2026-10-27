import sys
import hashlib
import io
import tempfile
import unittest
from contextlib import redirect_stderr
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from course_calendar_rag.ingest import (  # noqa: E402
    assert_allowed_url,
    fetch_and_parse,
    main,
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
        for item in self.passages:
            self.assertEqual(item.content_sha256, hashlib.sha256(item.text.encode()).hexdigest())
            self.assertEqual(
                item.id,
                "mcmaster-" + hashlib.sha256(f"{item.source}\n{item.section}\n{item.text}".encode()).hexdigest()[:16],
            )

    def test_short_sections_and_nested_items(self) -> None:
        html = """<main><h2>A</h2><p>Yes.</p><h2>B</h2>
        <ul><li>Parent <strong>rule</strong><ul><li>Child rule.</li></ul></li></ul></main>"""
        passages = parse_calendar_html(html, "https://academiccalendars.romcmaster.ca/page", "now")
        self.assertEqual([item.text for item in passages], ["Yes.", "Parent rule Child rule."])
        self.assertEqual(passages[1].text.count("Child rule."), 1)

    def test_same_text_has_same_content_hash_and_distinct_section_ids(self) -> None:
        html = "<main><h2>A</h2><p>Same.</p><h2>B</h2><p>Same.</p></main>"
        passages = parse_calendar_html(html, "https://academiccalendars.romcmaster.ca/page", "now")
        self.assertEqual(len(passages), 2)
        self.assertEqual(passages[0].content_sha256, passages[1].content_sha256)
        self.assertNotEqual(passages[0].id, passages[1].id)

    def test_empty_page_and_empty_output_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_calendar_html("<main><h2>Empty</h2></main>", "https://academiccalendars.romcmaster.ca/page", "now")
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            output = Path(directory) / "empty.jsonl"
            with self.assertRaises(ValueError):
                write_jsonl([], output)
            self.assertFalse(output.exists())

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
        ) as sleep:
            with self.assertRaises(httpx.HTTPStatusError):
                fetch_and_parse("https://academiccalendars.romcmaster.ca/calendar", delay_seconds=0)
        sleep.assert_called_once_with(1.0)
        self.assertEqual(
            requests,
            [
                "https://academiccalendars.romcmaster.ca/robots.txt",
                "https://academiccalendars.romcmaster.ca/calendar",
            ],
        )

    def test_crawl_delay_and_client_timestamp(self) -> None:
        html = "<main><h2>Rule</h2><p>Yes.</p></main>"

        def handler(request):
            if request.url.path == "/robots.txt":
                return httpx.Response(200, text="User-agent: *\nAllow: /\nCrawl-delay: 5")
            return httpx.Response(200, text=html, headers={"Date": "Thu, 01 Jan 1970 00:00:00 GMT"})

        client = httpx.Client(transport=httpx.MockTransport(handler))
        before = datetime.now().astimezone()
        with patch("course_calendar_rag.ingest.httpx.Client", return_value=client), patch(
            "course_calendar_rag.ingest.time.sleep"
        ) as sleep:
            passages = fetch_and_parse("https://academiccalendars.romcmaster.ca/page", delay_seconds=0)
        after = datetime.now().astimezone()
        sleep.assert_called_once_with(5.0)
        stamp = datetime.fromisoformat(passages[0].retrieved_at)
        self.assertEqual(stamp.utcoffset().total_seconds(), 0)
        self.assertLessEqual(before, stamp)
        self.assertLessEqual(stamp, after)

    def test_jsonl_writer(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            output = Path(directory) / "passages.jsonl"
            write_jsonl(self.passages, output)
            self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 2)

            original = output.read_bytes()
            with self.assertRaises(FileExistsError):
                write_jsonl(self.passages, output)
            self.assertEqual(output.read_bytes(), original)

    def test_existing_output_fails_before_any_request(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            output = Path(directory) / "approved.jsonl"
            output.write_bytes(b"approved export\n")
            url = "https://academiccalendars.romcmaster.ca/calendar"
            stderr = io.StringIO()
            with patch.object(sys, "argv", ["calendar-ingest", url, "--output", str(output)]), patch(
                "course_calendar_rag.ingest.fetch_and_parse"
            ) as fetch, redirect_stderr(stderr), self.assertRaises(SystemExit) as error:
                main()
            self.assertEqual(error.exception.code, 2)
            self.assertIn("output already exists", stderr.getvalue())
            fetch.assert_not_called()
            self.assertEqual(output.read_bytes(), b"approved export\n")


if __name__ == "__main__":
    unittest.main()
