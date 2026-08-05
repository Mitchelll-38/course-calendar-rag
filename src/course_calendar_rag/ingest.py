"""Permission-aware ingestion for official academic-calendar HTML."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

from bs4 import BeautifulSoup
import httpx


ALLOWED_HOSTS = {"academiccalendars.romcmaster.ca"}
USER_AGENT = "course-calendar-rag-research/0.1 (+https://github.com/Mitchelll-38/course-calendar-rag)"


@dataclass(frozen=True)
class IngestedPassage:
    id: str
    title: str
    text: str
    source: str
    section: str
    document_type: str
    retrieved_at: str
    content_sha256: str
    is_synthetic: bool = False


def assert_allowed_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS:
        raise ValueError(f"URL must use HTTPS on an allowlisted host: {sorted(ALLOWED_HOSTS)}")


def robots_allows(url: str, client: httpx.Client) -> bool:
    assert_allowed_url(url)
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    response = client.get(robots_url, headers={"User-Agent": USER_AGENT})
    if response.status_code >= 400:
        return False
    parser = RobotFileParser()
    parser.set_url(robots_url)
    parser.parse(response.text.splitlines())
    return parser.can_fetch(USER_AGENT, url)


def clean_text(element) -> str:
    return " ".join(element.get_text(" ", strip=True).split())


def parse_calendar_html(html: str, source_url: str, retrieved_at: str) -> list[IngestedPassage]:
    """Extract heading-scoped passages from Modern Campus-style calendar HTML."""
    assert_allowed_url(source_url)
    soup = BeautifulSoup(html, "html.parser")
    main = soup.select_one("#acalog-content, main, [role='main']")
    if main is None:
        raise ValueError("calendar content container was not found")

    page_title = clean_text(soup.select_one("h1")) if soup.select_one("h1") else "Academic calendar"
    passages: list[IngestedPassage] = []
    current_heading = page_title
    buffer: list[str] = []

    def flush() -> None:
        if not buffer:
            return
        text = " ".join(buffer).strip()
        buffer.clear()
        if len(text) < 40:
            return
        digest = hashlib.sha256(f"{source_url}\n{current_heading}\n{text}".encode()).hexdigest()
        passages.append(
            IngestedPassage(
                id=f"mcmaster-{digest[:16]}",
                title=page_title,
                text=text,
                source=source_url,
                section=current_heading,
                document_type="calendar",
                retrieved_at=retrieved_at,
                content_sha256=digest,
            )
        )

    for element in main.find_all(["h2", "h3", "h4", "p", "li"], recursive=True):
        if element.name in {"h2", "h3", "h4"}:
            flush()
            current_heading = clean_text(element)
        elif element.find_parent("li") is None or element.name == "li":
            text = clean_text(element)
            if text:
                buffer.append(text)
    flush()

    unique: dict[str, IngestedPassage] = {}
    for passage in passages:
        unique.setdefault(passage.content_sha256, passage)
    return list(unique.values())


def fetch_and_parse(url: str, *, delay_seconds: float = 2.0) -> list[IngestedPassage]:
    assert_allowed_url(url)
    with httpx.Client(timeout=20, follow_redirects=True) as client:
        if not robots_allows(url, client):
            raise PermissionError("robots.txt is unavailable or does not permit this user agent")
        time.sleep(delay_seconds)
        response = client.get(url, headers={"User-Agent": USER_AGENT})
        response.raise_for_status()
        retrieved_at = response.headers.get("date", "unknown")
        return parse_calendar_html(response.text, str(response.url), retrieved_at)


def write_jsonl(passages: list[IngestedPassage], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for passage in passages:
            handle.write(json.dumps(asdict(passage), ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest an explicitly permitted McMaster calendar page")
    parser.add_argument("url", help="Allowlisted HTTPS calendar URL")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--delay", type=float, default=2.0)
    args = parser.parse_args()
    passages = fetch_and_parse(args.url, delay_seconds=max(args.delay, 1.0))
    write_jsonl(passages, args.output)
    print(f"Wrote {len(passages)} passages to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
