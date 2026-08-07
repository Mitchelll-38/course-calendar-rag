"""Tokenization utilities with explicit support for academic course codes."""

from __future__ import annotations

import re


COURSE_CODE = re.compile(
    r"\b([A-Z]{2,}(?:\s+[A-Z]{2,})?)\s*[- ]?\s*(\d[A-Z]\d{2})\b",
)
TOKEN = re.compile(r"[a-z0-9]+")


def normalize_course_codes(text: str) -> str:
    """Join a spaced department name and course number into one search token."""

    def replace(match: re.Match[str]) -> str:
        department = re.sub(r"\s+", "", match.group(1))
        return f"{department}{match.group(2)}"

    return COURSE_CODE.sub(replace, text)


def tokenize(text: str) -> list[str]:
    normalized = normalize_course_codes(text).lower()
    return TOKEN.findall(normalized)
