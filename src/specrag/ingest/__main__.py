"""Build the index: python -m specrag.ingest [--parts K] [--stub]"""

from __future__ import annotations

import argparse
import time

from specrag.config import INDEX_DIR, RAW_DIR
from specrag.index.store import VectorStore
from specrag.ingest.chunk import pack_chunks, split_clauses
from specrag.ingest.fetch import discover, download
from specrag.ingest.parse import read_pdf


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts", nargs="*", help="limit to these parts, e.g. --parts K M")
    parser.add_argument("--stub", action="store_true", help="hashing embedder, no model download")
    parser.add_argument("--skip-download", action="store_true", help="use PDFs already in data/raw")
    args = parser.parse_args(argv)

    if args.skip_download:
        paths = sorted(RAW_DIR.glob("*.pdf"))
        print(f"using {len(paths)} PDF(s) already in {RAW_DIR}")
    else:
        pdfs = discover()
        if args.parts:
            wanted = {p.upper() for p in args.parts}
            pdfs = [p for p in pdfs if p.part in wanted]
        print(f"discovered {len(pdfs)} PDF(s)")
        paths = download(pdfs)

    chunks = []
    for path in paths:
        part = path.stem.split("-")[1] if "-" in path.stem else "?"
        pages = read_pdf(path)
        clauses = split_clauses(pages, part)
        part_chunks = pack_chunks(clauses, part)
        chunks.extend(part_chunks)
        print(
            f"  {path.name}: {len(pages)} pages, {len(clauses)} clauses, {len(part_chunks)} chunks"
        )

    if not chunks:
        print("no chunks produced; nothing to index")
        return 1

    if args.stub:
        from specrag.index.embedder import HashingEmbedder

        embedder = HashingEmbedder()
    else:
        from specrag.index.embedder import SentenceTransformerEmbedder

        embedder = SentenceTransformerEmbedder()

    started = time.time()
    store = VectorStore.build(chunks, embedder)
    store.save(INDEX_DIR)
    print(f"indexed {len(store)} chunks in {time.time() - started:.1f}s -> {INDEX_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
