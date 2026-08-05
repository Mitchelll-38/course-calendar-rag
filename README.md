# Course Calendar RAG

A hybrid retrieval question-answering system for McMaster University academic regulations and course-calendar content.

The project will combine lexical and semantic retrieval, reciprocal rank fusion, reranking, and a labeled evaluation suite. Development is organized as small, reviewable pull requests.

## Status

The first development milestone adds a small synthetic test corpus and labeled retrieval questions before any live calendar ingestion is introduced. See [`data/sample`](data/sample) for the dataset and its limitations.

## Validate the sample data

```shell
python scripts/validate_sample_data.py
python -m unittest discover -s tests -v
```

## BM25 baseline

The first retriever is a dependency-free BM25 implementation with course-code normalization. It provides a reproducible lexical baseline for later dense and hybrid retrieval experiments.

```shell
$env:PYTHONPATH="src" # PowerShell; use `export PYTHONPATH=src` on macOS/Linux
python -m course_calendar_rag.cli search "Can I take COMP SCI 2C03?"
python -m course_calendar_rag.cli evaluate --output results/bm25_baseline.json
```
