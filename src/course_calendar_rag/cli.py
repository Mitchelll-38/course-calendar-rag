"""Command-line interface for search and evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .bm25 import BM25Index
from .dense import DenseIndex
from .evaluation import evaluate, summary_as_dict
from .hybrid import HybridIndex


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PASSAGES = ROOT / "data" / "sample" / "passages.jsonl"
DEFAULT_QUESTIONS = ROOT / "data" / "sample" / "eval_questions.jsonl"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Course calendar retrieval baseline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    search = subparsers.add_parser("search", help="search the sample corpus")
    search.add_argument("query")
    search.add_argument("--top-k", type=int, default=5)
    search.add_argument("--retriever", choices=("bm25", "dense", "hybrid"), default="bm25")

    evaluation = subparsers.add_parser("evaluate", help="evaluate retrieval")
    evaluation.add_argument("--top-k", type=int, default=5)
    evaluation.add_argument("--output", type=Path)
    evaluation.add_argument("--retriever", choices=("bm25", "dense", "hybrid"), default="bm25")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    passages = load_jsonl(DEFAULT_PASSAGES)
    if args.retriever == "dense":
        index = DenseIndex(passages)
    elif args.retriever == "hybrid":
        index = HybridIndex(passages)
    else:
        index = BM25Index(passages)
    if args.command == "search":
        for rank, result in enumerate(index.search(args.query, top_k=args.top_k), start=1):
            print(f"{rank}. {result.passage_id} ({result.score:.4f}) — {result.passage['title']}")
        return 0

    summary, details = evaluate(index, load_jsonl(DEFAULT_QUESTIONS), top_k=args.top_k)
    payload = {"retriever": args.retriever, "summary": summary_as_dict(summary), "questions": details}
    rendered = json.dumps(payload, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Wrote evaluation results to {args.output}")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
