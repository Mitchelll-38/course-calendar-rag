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
python -m course_calendar_rag.cli evaluate --output /tmp/bm25_baseline.json
```

## Dense baseline

Dense retrieval currently uses a deterministic feature-hashing baseline. It runs locally with no model download:

```shell
python -m course_calendar_rag.cli search "Can I take COMP SCI 2C03?" --retriever dense
python -m course_calendar_rag.cli evaluate --retriever dense --output /tmp/dense_baseline.json
```

## Hybrid retrieval

Hybrid retrieval combines BM25 and dense results using weighted reciprocal rank fusion. It ranks passages by their positions in each result list, so the different score scales do not need to match.

```shell
python -m course_calendar_rag.cli search "Can I take COMP SCI 2C03?" --retriever hybrid
python -m course_calendar_rag.cli evaluate --retriever hybrid --output /tmp/hybrid_baseline.json
```

On the 20 synthetic questions, this default hybrid baseline has recall@5 of 0.875, compared with 0.925 for BM25 alone. Fusion does not guarantee better retrieval; use the labeled evaluation to assess weight and candidate settings before relying on it.

## Compare retrieval results

Run both retrievers on the same synthetic questions and inspect aggregate metric changes and per-question gains, losses, and gold-passage ranks:

```shell
python -m course_calendar_rag.cli compare --output /tmp/bm25_vs_hybrid.json
python -m course_calendar_rag.cli compare --top-k 10
```

The comparison uses the local feature-hashing dense baseline; no model download or live course data is needed. A negative delta means hybrid scored lower than BM25. `--top-k` must be at least five and sets retrieval depth, which can affect hybrid ranking; recall changes are measured at five.
The command refuses to overwrite an existing output file.
