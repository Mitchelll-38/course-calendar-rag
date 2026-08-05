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

    def test_course_code_queries_are_represented(self) -> None:
        categories = {question["category"] for question in self.questions}
        self.assertIn("course_lookup", categories)

    def test_every_passage_is_explicitly_synthetic(self) -> None:
        self.assertTrue(all(passage["is_synthetic"] for passage in self.passages))


if __name__ == "__main__":
    unittest.main()
