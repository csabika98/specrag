"""Tests for clause splitting and chunk packing."""

from __future__ import annotations

from specrag.ingest.chunk import Chunk, pack_chunks, split_clauses
from specrag.ingest.parse import Page


def page(number: int, *lines: str) -> Page:
    return Page(number=number, text="\n".join(lines))


class TestSplitClauses:
    def test_extracts_number_and_text(self):
        [clause] = split_clauses([page(1, "1.1 The guidance covers stairs.")], "K")
        assert clause.number == "1.1"
        assert clause.text == "The guidance covers stairs."
        assert clause.page == 1

    def test_joins_lines_that_wrap_within_a_clause(self):
        pages = [
            page(
                1,
                "1.1 The guidance covers internal and external steps",
                "when part of the building.",
            )
        ]
        [clause] = split_clauses(pages, "K")
        assert (
            clause.text
            == "The guidance covers internal and external steps when part of the building."
        )

    def test_continues_a_clause_across_a_page_break(self):
        pages = [
            page(1, "1.1 The guidance covers steps"),
            page(2, "that form part of the building."),
        ]
        [clause] = split_clauses(pages, "K")
        assert clause.text == "The guidance covers steps that form part of the building."
        assert clause.page == 1

    def test_separates_consecutive_clauses(self):
        pages = [page(1, "1.1 First clause here.", "1.2 Second clause here.")]
        assert [c.number for c in split_clauses(pages, "K")] == ["1.1", "1.2"]

    def test_attaches_a_heading_that_introduces_a_clause(self):
        pages = [page(1, "Steepness of stairs", "1.2 Measure the rise and going.")]
        [clause] = split_clauses(pages, "K")
        assert clause.heading == "Steepness of stairs"

    def test_does_not_treat_diagram_labels_as_a_heading(self):
        # Extraction pulls loose labels out of diagrams. They are followed by
        # more labels, never by a clause, which is how they are told apart.
        pages = [page(1, "1.1 Real clause text here.", "Going", "Open riser", "Nosing")]
        [clause] = split_clauses(pages, "K")
        assert clause.heading is None
        assert "Going" in clause.text  # kept as body, not promoted to heading

    def test_a_heading_carries_to_later_clauses_in_the_same_section(self):
        pages = [page(1, "Stair dimensions", "1.2 First.", "1.3 Second.")]
        assert [c.heading for c in split_clauses(pages, "K")] == ["Stair dimensions"] * 2

    def test_ignores_a_decimal_measurement_at_the_start_of_a_line(self):
        # "2.5 m clear width" must not be read as clause 2.5.
        pages = [page(1, "1.1 The width is as follows.", "2.5 m clear width is required.")]
        [clause] = split_clauses(pages, "K")
        assert clause.number == "1.1"
        assert "2.5 m clear width" in clause.text

    def test_ignores_a_cross_reference_inside_a_sentence(self):
        pages = [page(1, "1.1 For tapered treads see paragraphs 1.25-1.27 below.")]
        [clause] = split_clauses(pages, "K")
        assert clause.number == "1.1"

    def test_drops_text_before_the_first_clause(self):
        pages = [page(1, "Some front matter prose that ends with a period.", "1.1 Real clause.")]
        [clause] = split_clauses(pages, "K")
        assert clause.text == "Real clause."

    def test_returns_empty_for_pages_with_no_clauses(self):
        assert split_clauses([page(1, "Contents", "Appendix A")], "K") == []


class TestPackChunks:
    def _clauses(self, count: int, size: int, heading: str | None = "H"):
        from specrag.ingest.chunk import Clause

        return [
            Clause(number=f"1.{i + 1}", heading=heading, text="x" * size, page=1)
            for i in range(count)
        ]

    def test_merges_short_clauses_into_one_chunk(self):
        chunks = pack_chunks(self._clauses(4, 60), "K", max_chars=1000, min_chars=200)
        assert len(chunks) == 1
        assert chunks[0].clauses == ("1.1", "1.2", "1.3", "1.4")

    def test_does_not_merge_across_a_heading_boundary(self):
        from specrag.ingest.chunk import Clause

        clauses = [
            Clause(number="1.1", heading="Stairs", text="a" * 60, page=1),
            Clause(number="2.1", heading="Ladders", text="b" * 60, page=2),
        ]
        chunks = pack_chunks(clauses, "K", max_chars=1000, min_chars=500)
        assert [c.heading for c in chunks] == ["Stairs", "Ladders"]

    def test_splits_a_clause_longer_than_the_limit(self):
        from specrag.ingest.chunk import Clause

        long_text = " ".join(f"Sentence number {i} runs on." for i in range(60))
        chunks = pack_chunks(
            [Clause(number="1.1", heading=None, text=long_text, page=1)],
            "K",
            max_chars=300,
            min_chars=100,
        )
        assert len(chunks) > 1
        assert all(len(c.text) <= 400 for c in chunks)

    def test_records_the_page_range_it_spans(self):
        from specrag.ingest.chunk import Clause

        clauses = [
            Clause(number="1.1", heading="H", text="a" * 60, page=3),
            Clause(number="1.2", heading="H", text="b" * 60, page=5),
        ]
        [chunk] = pack_chunks(clauses, "K", max_chars=1000, min_chars=100)
        assert (chunk.page_start, chunk.page_end) == (3, 5)

    def test_chunk_ids_are_unique(self):
        chunks = pack_chunks(self._clauses(20, 150), "K", max_chars=400, min_chars=200)
        assert len({c.chunk_id for c in chunks}) == len(chunks)

    def test_returns_empty_for_no_clauses(self):
        assert pack_chunks([], "K") == []


class TestChunkPresentation:
    def test_citation_names_a_single_paragraph(self):
        chunk = Chunk("id", "K", None, ("1.2",), "t", 1, 1)
        assert chunk.citation == "Approved Document K, paragraph 1.2"

    def test_citation_names_a_paragraph_range(self):
        chunk = Chunk("id", "K", None, ("1.2", "1.3", "1.4"), "t", 1, 1)
        assert chunk.citation == "Approved Document K, paragraphs 1.2-1.4"

    def test_embedding_text_includes_the_heading(self):
        # "Measure the rise and going" never says "stairs"; the heading does.
        chunk = Chunk("id", "K", "Steepness of stairs", ("1.2",), "Measure the rise.", 1, 1)
        assert chunk.to_embedding_text() == "Steepness of stairs\nMeasure the rise."

    def test_embedding_text_without_a_heading_is_just_the_text(self):
        chunk = Chunk("id", "K", None, ("1.2",), "Measure the rise.", 1, 1)
        assert chunk.to_embedding_text() == "Measure the rise."


class TestChunkIds:
    def test_a_single_clause_chunk_is_named_after_that_clause(self):
        from specrag.ingest.chunk import Clause

        [chunk] = pack_chunks(
            [Clause(number="1.2", heading=None, text="x" * 300, page=1)], "K", 1000, 200
        )
        assert chunk.chunk_id == "K:1.2"

    def test_a_merged_chunk_is_named_for_its_span(self):
        from specrag.ingest.chunk import Clause

        clauses = [
            Clause(number="1.1", heading="H", text="a" * 100, page=1),
            Clause(number="1.2", heading="H", text="b" * 100, page=1),
        ]
        [chunk] = pack_chunks(clauses, "K", 1000, 150)
        assert chunk.chunk_id == "K:1.1-1.2"

    def test_split_pieces_are_suffixed_rather_than_repeating_the_clause(self):
        from specrag.ingest.chunk import Clause

        long_text = " ".join(f"Sentence {i} here." for i in range(80))
        chunks = pack_chunks(
            [Clause(number="0.7", heading=None, text=long_text, page=1)], "K", 300, 100
        )
        ids = [c.chunk_id for c in chunks]
        assert ids[0] == "K:0.7"
        assert ids[1] == "K:0.7#2"
