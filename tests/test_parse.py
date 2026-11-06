"""Tests for PDF text cleaning. Pure functions, no PDF needed."""

from __future__ import annotations

from specrag.ingest.parse import (
    clean_page,
    is_running_header,
    normalise,
    strip_watermark,
)


class TestNormalise:
    def test_folds_ligatures(self):
        # Without this, a query for "profiles" never matches the typeset text.
        assert normalise("proﬁles and ﬂoor and ofﬁce") == "profiles and floor and office"

    def test_straightens_quotes_and_dashes(self):
        assert normalise("‘a’ “b” 1–2") == "'a' \"b\" 1-2"

    def test_collapses_runs_of_spaces(self):
        assert normalise("a    b\tc") == "a b c"

    def test_keeps_newlines_because_clause_structure_depends_on_them(self):
        assert normalise("1.1 First\n1.2 Second") == "1.1 First\n1.2 Second"

    def test_replaces_non_breaking_space(self):
        assert normalise("a\xa0b") == "a b"


class TestStripWatermark:
    def test_removes_every_occurrence(self):
        assert strip_watermark("aONLINE VERSIONbONLINE VERSIONc") == "abc"

    def test_removes_a_copy_truncated_by_the_page_boundary(self):
        # Extraction frequently ends mid-watermark.
        assert strip_watermark("real text ONLINE VER").strip() == "real text"

    def test_leaves_text_without_a_watermark_alone(self):
        assert strip_watermark("1.1 The guidance") == "1.1 The guidance"


class TestIsRunningHeader:
    def test_detects_both_recto_and_verso_header_layouts(self):
        assert is_running_header("Building Regulations 2010 Approved Document K, 2013 edition 3")
        assert is_running_header("4 Approved Document K, 2013 edition Building Regulations 2010")

    def test_detects_a_bare_page_number(self):
        assert is_running_header("  12 ")

    def test_does_not_discard_body_text(self):
        assert not is_running_header("1.1 The guidance provided in this document covers stairs.")
        assert not is_running_header("See Approved Document M Section 1 for stepped access.")

    def test_blank_lines_are_not_headers(self):
        assert not is_running_header("   ")


class TestCleanPage:
    def test_strips_header_and_watermark_but_keeps_the_clause(self):
        raw = (
            "4 Approved Document K, 2013 edition Building Regulations 2010\n"
            "ONLINE VERSION\n"
            "1.1 The guidance provided in this document covers stairs.\n"
            "ONLINE VERSION\n"
        )
        cleaned = clean_page(raw)
        assert "Approved Document K, 2013 edition" not in cleaned
        assert "ONLINE VERSION" not in cleaned
        assert cleaned == "1.1 The guidance provided in this document covers stairs."

    def test_keeps_a_cross_reference_to_another_part(self):
        # "Approved Document M" without an edition is body text, not a header.
        raw = "1.1 See Approved Document M Section 1 for stepped access."
        assert "Approved Document M" in clean_page(raw)

    def test_returns_empty_for_a_page_that_was_only_furniture(self):
        raw = "ONLINE VERSION\n12\nBuilding Regulations 2010 Approved Document K, 2013 edition 3\n"
        assert clean_page(raw) == ""


class TestRunningHeaderFragments:
    """The PDF content stream splits the header mid-word, so extraction
    produces partial and doubled variants. These are what actually appear."""

    def test_detects_a_header_fragment_missing_its_opening_words(self):
        assert is_running_header("t K, 2013 edition")

    def test_detects_a_doubled_header_fragment(self):
        assert is_running_header("t K, 2013 editiont K, 2013 edition")

    def test_keeps_cover_text_that_mentions_the_edition(self):
        assert not is_running_header("2013 edition - for use in England*")

    def test_keeps_a_contents_heading_that_mentions_the_edition(self):
        assert not is_running_header("Main changes in the 2013 edition")

    def test_keeps_a_sentence_that_happens_to_cite_an_edition(self):
        line = "1.4 This replaces the guidance in Approved Document K, 2013 edition, for stairs."
        assert not is_running_header(line)
