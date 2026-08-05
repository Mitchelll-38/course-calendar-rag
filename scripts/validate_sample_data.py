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
    passage_ids = [record.get("id") for record in passages]
    question_ids = [record.get("id") for record in questions]

    if len(passage_ids) != len(set(passage_ids)):
        errors.append("passage IDs must be unique")
    if len(question_ids) != len(set(question_ids)):
        errors.append("question IDs must be unique")

    for record in passages:
        missing = PASSAGE_FIELDS - record.keys()
        if missing:
            errors.append(f"passage {record.get('id', '<unknown>')} missing {sorted(missing)}")
        if record.get("is_synthetic") is not True:
            errors.append(f"passage {record.get('id', '<unknown>')} must be marked synthetic")

    known_passage_ids = set(passage_ids)
    for record in questions:
        missing = QUESTION_FIELDS - record.keys()
        if missing:
            errors.append(f"question {record.get('id', '<unknown>')} missing {sorted(missing)}")
        gold_ids = record.get("gold_passage_ids", [])
        if not isinstance(gold_ids, list) or not gold_ids:
            errors.append(f"question {record.get('id', '<unknown>')} needs gold passage IDs")
        else:
            unknown = set(gold_ids) - known_passage_ids
            if unknown:
                errors.append(
                    f"question {record.get('id', '<unknown>')} references unknown passages {sorted(unknown)}"
                )
        if record.get("difficulty") not in DIFFICULTIES:
            errors.append(f"question {record.get('id', '<unknown>')} has invalid difficulty")

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
