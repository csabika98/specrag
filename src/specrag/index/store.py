"""FAISS-backed vector store.

A flat inner-product index over unit-norm vectors, which makes the score
cosine similarity. The corpus is a few thousand chunks, so an approximate
index would add tuning and failure modes for no measurable gain.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from specrag.index.embedder import Embedder
from specrag.ingest.chunk import Chunk

_INDEX_FILE = "chunks.faiss"
_META_FILE = "chunks.jsonl"


class VectorStore:
    def __init__(self, index, chunks: list[Chunk]):
        self.index = index
        self.chunks = chunks

    def __len__(self) -> int:
        return len(self.chunks)

    @classmethod
    def build(cls, chunks: list[Chunk], embedder: Embedder) -> VectorStore:
        import faiss

        index = faiss.IndexFlatIP(embedder.dimension)
        if chunks:
            vectors = embedder.encode([c.to_embedding_text() for c in chunks])
            index.add(vectors)
        return cls(index=index, chunks=list(chunks))

    def search(self, query: str, embedder: Embedder, k: int = 5) -> list[tuple[Chunk, float]]:
        """Return the k nearest chunks with their cosine scores."""
        if not self.chunks:
            return []
        k = min(k, len(self.chunks))
        scores, indices = self.index.search(embedder.encode([query]), k)
        hits = []
        for position, score in zip(indices[0], scores[0], strict=True):
            if position < 0:  # faiss pads with -1 when fewer results exist
                continue
            hits.append((self.chunks[int(position)], float(score)))
        return hits

    def save(self, directory: Path) -> None:
        import faiss

        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / _INDEX_FILE))
        with (directory / _META_FILE).open("w", encoding="utf-8") as handle:
            for chunk in self.chunks:
                handle.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, directory: Path) -> VectorStore:
        import faiss

        index = faiss.read_index(str(directory / _INDEX_FILE))
        chunks = []
        with (directory / _META_FILE).open(encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                payload["clauses"] = tuple(payload["clauses"])
                chunks.append(Chunk(**payload))
        if index.ntotal != len(chunks):
            raise ValueError(
                f"Index holds {index.ntotal} vectors but metadata has {len(chunks)} chunks; "
                f"rebuild the index in {directory}"
            )
        return cls(index=index, chunks=chunks)
