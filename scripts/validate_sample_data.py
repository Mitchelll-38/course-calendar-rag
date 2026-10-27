"""Validate the checked-in synthetic corpus and retrieval evaluation set."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PASSAGES_PATH = ROOT / "data" / "sample" / "passages.jsonl"
QUESTIONS_PATH = ROOT / "data" / "sample" / "eval_questions.jsonl"

PASSAGE_FIELDS = {
    "id",
    "title",
    "text",
    "source",
    "section",
    "document_type",
    "faculty",
    "academic_year",
    "is_synthetic",
}
QUESTION_FIELDS = {"id", "question", "gold_passage_ids", "category", "difficulty"}
DIFFICULTIES = {"easy", "medium", "hard"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(record)
    return records


def validate_records(
    passages: list[dict[str, Any]], questions: list[dict[str, Any]]
) -> list[str]:
    errors: list[str] = []
    if not passages:
        errors.append("at least one passage is required")
    if not questions:
        errors.append("at least one evaluation question is required")
    passage_ids: set[str] = set()
    question_ids: set[str] = set()

    for row, record in enumerate(passages, start=1):
        label = f"passage row {row}"
        if not isinstance(record, dict):
            errors.append(f"{label} must be a JSON object")
            continue
        missing = PASSAGE_FIELDS - record.keys()
        if missing:
            errors.append(f"{label} missing {sorted(missing)}")
        for field in sorted(PASSAGE_FIELDS - {"is_synthetic"}):
            value = record.get(field)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{label} {field} must be a nonempty string")
        passage_id = record.get("id")
        if isinstance(passage_id, str) and passage_id.strip():
            if passage_id in passage_ids:
                errors.append(f"duplicate passage ID: {passage_id}")
            passage_ids.add(passage_id)
        source = record.get("source")
        if isinstance(source, str) and source.strip():
            if not source.startswith("synthetic://"):
                errors.append(f"{label} source must start with synthetic://")
            elif not source.removeprefix("synthetic://").strip():
                errors.append(f"{label} source must identify a synthetic document")
        if record.get("is_synthetic") is not True:
            errors.append(f"{label} must be marked synthetic")

    for row, record in enumerate(questions, start=1):
        label = f"question row {row}"
        if not isinstance(record, dict):
            errors.append(f"{label} must be a JSON object")
            continue
        missing = QUESTION_FIELDS - record.keys()
        if missing:
            errors.append(f"{label} missing {sorted(missing)}")
        for field in sorted(QUESTION_FIELDS - {"gold_passage_ids", "difficulty"}):
            value = record.get(field)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{label} {field} must be a nonempty string")
        question_id = record.get("id")
        if isinstance(question_id, str) and question_id.strip():
            if question_id in question_ids:
                errors.append(f"duplicate question ID: {question_id}")
            question_ids.add(question_id)
        gold_ids = record.get("gold_passage_ids", [])
        if not isinstance(gold_ids, list) or not gold_ids:
            errors.append(f"{label} needs a nonempty list of gold passage IDs")
        else:
            seen: set[str] = set()
            for gold_id in gold_ids:
                if not isinstance(gold_id, str) or not gold_id.strip():
                    errors.append(f"{label} gold passage IDs must be nonempty strings")
                    continue
                if gold_id in seen:
                    errors.append(f"{label} repeats gold passage ID: {gold_id}")
                elif gold_id not in passage_ids:
                    errors.append(f"{label} references unknown passage: {gold_id}")
                seen.add(gold_id)
        difficulty = record.get("difficulty")
        if not isinstance(difficulty, str) or difficulty not in DIFFICULTIES:
            errors.append(f"{label} has invalid difficulty")

    return errors


def main() -> int:
    passages = load_jsonl(PASSAGES_PATH)
    questions = load_jsonl(QUESTIONS_PATH)
    errors = validate_records(passages, questions)
    if errors:
        print("Sample data validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"Validated {len(passages)} passages and {len(questions)} evaluation questions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
