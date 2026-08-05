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

## Dense baseline

Dense retrieval supports a deterministic feature-hashing baseline with no model download, plus an optional Sentence Transformers adapter:

```shell
python -m course_calendar_rag.cli evaluate --retriever dense --output results/dense_baseline.json
pip install -e ".[dense]"
python -m course_calendar_rag.cli evaluate --retriever dense --sentence-transformer
```

## Hybrid retrieval

Weighted reciprocal rank fusion combines BM25 and dense rankings without requiring their raw scores to share a scale:

```shell
python -m course_calendar_rag.cli search "COMP SCI 2C03 prerequisite" --retriever hybrid
python -m course_calendar_rag.cli evaluate --retriever hybrid --output results/hybrid_baseline.json
```

## Reranking

An interpretable candidate reranker prioritizes title matches and exact course codes. It can later be replaced by a cross-encoder behind the same retrieval interface.

```shell
python -m course_calendar_rag.cli evaluate --retriever reranked --output results/reranked_baseline.json
```

## API

Install the API extra and start the service:

```shell
pip install -e ".[api]"
uvicorn course_calendar_rag.api:app --reload
```

Interactive documentation is available at `http://localhost:8000/docs`. The current answer endpoint is deliberately extractive and always returns ranked source passages with a synthetic-data disclaimer.

## Frontend

The Next.js interface lives in `frontend`:

```shell
cd frontend
npm install
npm run dev
```
