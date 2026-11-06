"""Tests for the vector store. Uses the hashing stub, so no model download."""

from __future__ import annotations

import subprocess
import sys

import pytest

from specrag.index.embedder import HashingEmbedder
from specrag.index.store import VectorStore
from specrag.ingest.chunk import Chunk


def chunk(cid: str, text: str, heading: str | None = None) -> Chunk:
    return Chunk(
        chunk_id=cid, part="K", heading=heading, clauses=(cid,), text=text, page_start=1, page_end=1
    )


@pytest.fixture
def embedder():
    return HashingEmbedder(dimension=128)


@pytest.fixture
def chunks():
    return [
        chunk("1.1", "The private stair rise should be a maximum of 220mm."),
        chunk("1.2", "Handrails should be provided on at least one side of the stair."),
        chunk("2.1", "Ladders may be used for access to a loft conversion."),
        chunk("3.1", "Vehicle barriers should resist horizontal loading."),
    ]


class TestHashingEmbedder:
    def test_produces_unit_norm_vectors(self, embedder):
        import numpy as np

        vectors = embedder.encode(["stair rise", "handrail height"])
        assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)

    def test_is_deterministic_within_a_process(self, embedder):
        import numpy as np

        assert np.array_equal(embedder.encode(["stairs"]), embedder.encode(["stairs"]))

    def test_is_deterministic_across_processes(self):
        # builtin hash() is salted per process; a salted stub would make an
        # index built in one run unusable by a query in the next.
        code = (
            "from specrag.index.embedder import HashingEmbedder;"
            "print(round(float(HashingEmbedder(32).encode(['stairs'])[0][3]), 6))"
        )
        runs = {
            subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                check=True,
                env={"PYTHONHASHSEED": seed, "PYTHONPATH": "src"},
            ).stdout.strip()
            for seed in ("0", "1", "12345")
        }
        assert len(runs) == 1

    def test_handles_an_empty_batch(self, embedder):
        assert embedder.encode([]).shape == (0, 128)


class TestVectorStore:
    def test_indexes_every_chunk(self, chunks, embedder):
        store = VectorStore.build(chunks, embedder)
        assert len(store) == 4
        assert store.index.ntotal == 4

    def test_search_returns_at_most_k_hits(self, chunks, embedder):
        store = VectorStore.build(chunks, embedder)
        assert len(store.search("stair rise", embedder, k=2)) == 2

    def test_search_caps_k_at_the_corpus_size(self, chunks, embedder):
        store = VectorStore.build(chunks, embedder)
        assert len(store.search("stair", embedder, k=99)) == 4

    def test_search_returns_scores_in_descending_order(self, chunks, embedder):
        store = VectorStore.build(chunks, embedder)
        scores = [score for _, score in store.search("stair rise maximum", embedder, k=4)]
        assert scores == sorted(scores, reverse=True)

    def test_search_on_an_empty_store_returns_nothing(self, embedder):
        assert VectorStore.build([], embedder).search("anything", embedder) == []

    def test_round_trips_through_disk(self, chunks, embedder, tmp_path):
        VectorStore.build(chunks, embedder).save(tmp_path)
        reloaded = VectorStore.load(tmp_path)
        assert len(reloaded) == 4
        assert [c.chunk_id for c in reloaded.chunks] == [c.chunk_id for c in chunks]

    def test_reloaded_store_returns_the_same_results(self, chunks, embedder, tmp_path):
        original = VectorStore.build(chunks, embedder)
        original.save(tmp_path)
        before = [c.chunk_id for c, _ in original.search("handrail", embedder, k=3)]
        after = [
            c.chunk_id for c, _ in VectorStore.load(tmp_path).search("handrail", embedder, k=3)
        ]
        assert before == after

    def test_preserves_chunk_metadata_across_a_reload(self, embedder, tmp_path):
        VectorStore.build([chunk("1.2", "text", heading="Stairs")], embedder).save(tmp_path)
        restored = VectorStore.load(tmp_path).chunks[0]
        assert restored.heading == "Stairs"
        assert restored.clauses == ("1.2",)
        assert restored.citation == "Approved Document K, paragraph 1.2"

    def test_rejects_an_index_that_disagrees_with_its_metadata(self, chunks, embedder, tmp_path):
        VectorStore.build(chunks, embedder).save(tmp_path)
        meta = tmp_path / "chunks.jsonl"
        meta.write_text("\n".join(meta.read_text().splitlines()[:2]) + "\n")
        with pytest.raises(ValueError, match="rebuild the index"):
            VectorStore.load(tmp_path)
