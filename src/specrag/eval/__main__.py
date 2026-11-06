"""Evaluate retrieval: python -m specrag.eval [--stub]"""

from __future__ import annotations

import argparse
import sys

from specrag.config import INDEX_DIR
from specrag.eval.harness import evaluate, format_failures, load_questions
from specrag.index.store import VectorStore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=5, help="retrieval depth (default 5)")
    parser.add_argument(
        "--stub",
        action="store_true",
        help="use the hashing embedder instead of the real model (no download)",
    )
    args = parser.parse_args(argv)

    if args.stub:
        from specrag.index.embedder import HashingEmbedder

        embedder = HashingEmbedder()
    else:
        from specrag.index.embedder import SentenceTransformerEmbedder

        embedder = SentenceTransformerEmbedder()

    if not (INDEX_DIR / "chunks.faiss").exists():
        print(f"No index at {INDEX_DIR}. Build one first.", file=sys.stderr)
        return 1

    store = VectorStore.load(INDEX_DIR)
    questions = load_questions()
    report, results = evaluate(store, embedder, questions, k=args.k)
    print(f"corpus: {len(store)} chunks\n")
    print(report.as_table())
    print()
    print(format_failures(results, k=args.k))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
