import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.validate_sample_data import (  # noqa: E402
    PASSAGES_PATH,
    QUESTIONS_PATH,
    load_jsonl,
    validate_records,
)


class SampleDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.passages = load_jsonl(PASSAGES_PATH)
        cls.questions = load_jsonl(QUESTIONS_PATH)

    def test_expected_sample_sizes(self) -> None:
        self.assertEqual(len(self.passages), 12)
        self.assertEqual(len(self.questions), 20)

    def test_dataset_is_valid(self) -> None:
        self.assertEqual(validate_records(self.passages, self.questions), [])

    def test_empty_fixture_is_rejected(self) -> None:
        self.assertIn("at least one passage is required", validate_records([], self.questions))
        self.assertIn("at least one evaluation question is required", validate_records(self.passages, []))
        self.assertEqual(len(validate_records([], [])), 2)

    def test_course_code_queries_are_represented(self) -> None:
        categories = {question["category"] for question in self.questions}
        self.assertIn("course_lookup", categories)

    def test_every_passage_is_explicitly_synthetic(self) -> None:
        self.assertTrue(all(passage["is_synthetic"] for passage in self.passages))

    def test_rejects_non_synthetic_source_even_when_flagged(self) -> None:
        passage = {**self.passages[0], "source": "https://example.com/calendar"}
        errors = validate_records([passage], [])
        self.assertTrue(any("source must start with synthetic://" in error for error in errors))
        passage["source"] = "synthetic://  "
        errors = validate_records([passage], [])
        self.assertTrue(any("source must identify a synthetic document" in error for error in errors))

    def test_rejects_blank_and_wrong_type_fields(self) -> None:
        for field, value in (("title", "  "), ("text", None), ("id", [])):
            with self.subTest(field=field):
                passage = {**self.passages[0], field: value}
                errors = validate_records([passage], [])
                self.assertTrue(any(f"{field} must be a nonempty string" in error for error in errors))

        question = {**self.questions[0], "question": " ", "category": 7}
        errors = validate_records(self.passages, [question])
        self.assertTrue(any("question must be a nonempty string" in error for error in errors))
        self.assertTrue(any("category must be a nonempty string" in error for error in errors))

    def test_reports_malformed_gold_ids_without_crashing(self) -> None:
        question = {**self.questions[0], "gold_passage_ids": [[], "missing", "missing", " "]}
        errors = validate_records(self.passages, [question])
        self.assertTrue(any("gold passage IDs must be nonempty strings" in error for error in errors))
        self.assertTrue(any("references unknown passage: missing" in error for error in errors))
        self.assertTrue(any("repeats gold passage ID: missing" in error for error in errors))

    def test_reports_duplicate_ids_and_invalid_difficulty(self) -> None:
        question = {**self.questions[0], "difficulty": []}
        errors = validate_records(
            [self.passages[0], self.passages[0]], [question, question]
        )
        self.assertTrue(any("duplicate passage ID" in error for error in errors))
        self.assertTrue(any("duplicate question ID" in error for error in errors))
        self.assertTrue(any("invalid difficulty" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
