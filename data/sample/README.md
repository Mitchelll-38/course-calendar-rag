# Synthetic sample dataset

This directory provides a deliberately small corpus for developing and testing retrieval before live McMaster calendar ingestion exists.

## Files

- `passages.jsonl`: 12 synthetic regulation and course passages with stable IDs and retrieval metadata.
- `eval_questions.jsonl`: 20 questions labeled with one or more relevant passage IDs.

Every passage uses a `synthetic://` source and sets `is_synthetic` to `true`. The text is illustrative, is not copied from the McMaster calendar, and must not be treated as current or authoritative academic advice.

## Validate

From the repository root:

```shell
python scripts/validate_sample_data.py
python -m unittest discover -s tests -v
```

The live-ingestion milestone will replace or supplement this fixture with source-attributed content and separate the test fixture from production indexes.
