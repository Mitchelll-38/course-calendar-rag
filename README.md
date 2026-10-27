# Course Calendar RAG

A retrieval question-answering research prototype for McMaster University academic regulations and course-calendar content.

The current prototype combines lexical retrieval, a deterministic feature-hashing dense baseline, reciprocal rank fusion, feature reranking, and a labeled synthetic evaluation suite.

## Status

The local CLI, API, and Next.js frontend run against a small synthetic corpus. The separate ingestion command can extract an explicitly permitted calendar page, but its output is not connected to the demo search index. See [`data/sample`](data/sample) for the fixture and its limitations, and [`docs/ingestion.md`](docs/ingestion.md) for access rules.

## Local quickstart

Install Python 3.11 or newer and Node.js 20.9.0 or newer with npm (required by the locked Next.js version). Run these commands from the repository root. The Python package has no required third-party dependencies for CLI search; install the optional `api` dependencies to use the browser demo.

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[api]"
python -m course_calendar_rag.cli search "Can I take COMP SCI 2C03?"
uvicorn course_calendar_rag.api:app --host 127.0.0.1 --port 8000 --no-access-log
```

**macOS/Linux shell:**

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[api]'
python -m course_calendar_rag.cli search "Can I take COMP SCI 2C03?"
uvicorn course_calendar_rag.api:app --host 127.0.0.1 --port 8000 --no-access-log
```

Keep that terminal running. In a **second terminal**, start the frontend (PowerShell and macOS/Linux):

```sh
cd frontend
npm ci
npm run dev
```

Open <http://localhost:3000> and try a sample question. The API documentation is at <http://127.0.0.1:8000/docs>; <http://127.0.0.1:8000/health> reports the dataset as `synthetic`. The frontend calls `http://localhost:8000` by default; `frontend/.env.example` shows the `NEXT_PUBLIC_API_URL` setting if you need to configure that address before starting Next.js. The API accepts browser requests only from `http://localhost:3000` and `http://127.0.0.1:3000`. If Next.js starts on another port, free port 3000 and restart it for this local setup.

For CLI-only use, install with `python -m pip install -e .` and skip both servers. This demo uses illustrative `synthetic://` passages, not live calendar content or authoritative academic advice. The dense option uses deterministic feature hashing, not a trained semantic embedding model. The bundled evaluation questions are synthetic and do not establish real-world performance.

Calendar ingestion is a separate, permission-gated workflow. The [`ingestion guide`](docs/ingestion.md) describes a URL fetch command to use only with explicit permission and an allowlisted page that permits the request. An approved export is another possible source, but no export-import command is provided here. Any ingested or exported content requires review and an additional indexing step before it could be searched; the local API and frontend continue to use the synthetic fixture.

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

## Search a local calendar export

After a calendar page has been fetched with permission and exported to JSONL by
`calendar-ingest`, search that approved **local file** with any CLI retriever:

```shell
python -m course_calendar_rag.cli search "registration deadline" --passages path/to/approved-passages.jsonl --retriever hybrid
```

The `--passages` option only reads the supplied file; search makes no network
requests. Each nonblank JSONL row must be an object with a nonempty string
`id`, `title`, and `text`, and IDs must be unique. Without this option, search
still uses the synthetic sample. `evaluate` and `compare` continue to use only
the synthetic labeled sample; their metrics do not assess an ingested export.

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

## Feature reranking

The `reranked` retriever takes the top 10 hybrid candidates and scores them using interpretable features: query-token overlap with the title and body, plus a bonus when the query contains the passage's exact course code. It returns the highest-scoring passages first. The score printed by `search` is the reranker score. This runs locally with the feature-hashing dense baseline and requires no model download.

```shell
python -m course_calendar_rag.cli search "Can I take COMP SCI 2C03?" --retriever reranked
python -m course_calendar_rag.cli evaluate --retriever reranked --output /tmp/reranked_baseline.json
```

On the 20 synthetic questions, the default reranked retriever has recall@5 of 0.975, versus 0.875 for hybrid. Its recall@1 is 0.55 and MRR is 0.775, both below hybrid's 0.75 and 0.85. Use the per-question results to inspect where reranking helps or hurts; these sample metrics do not establish performance on live calendar content.

## Compare retrieval results

Run both retrievers on the same synthetic questions and inspect aggregate metric changes and per-question gains, losses, and gold-passage ranks:

```shell
python -m course_calendar_rag.cli compare --output /tmp/bm25_vs_hybrid.json
python -m course_calendar_rag.cli compare --top-k 10
```

The comparison uses the local feature-hashing dense baseline; no model download or live course data is needed. A negative delta means hybrid scored lower than BM25. `--top-k` must be at least five and sets retrieval depth, which can affect hybrid ranking; recall changes are measured at five.
The command refuses to overwrite an existing output file.

## Local API demo

From the repository root, install the optional API dependencies and start the local service:

```shell
pip install -e '.[api]'
uvicorn course_calendar_rag.api:app --host 127.0.0.1 --port 8000 --no-access-log
```

Open <http://127.0.0.1:8000/docs> for the interactive API documentation. The `/answer` endpoint returns an extractive answer from the top retrieved passage with a disclaimer. The service reads only the synthetic fixture in the source tree. It does not ingest live calendar data, and its answers are not authoritative academic advice.

Browser requests are allowed from the frontend development origins `http://localhost:3000` and `http://127.0.0.1:3000` through CORS. This is a source-tree demo: public deployment and packaging the fixture as installed data are not configured.
Use only non-sensitive demo queries. The GET `/search` route places the query in the URL, which clients or proxies may log; POST `/answer` sends it in the request body. CORS is not authentication.
