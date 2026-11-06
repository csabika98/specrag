"""Tests for corpus discovery. All offline: the API payloads are fixtures."""

from __future__ import annotations

import pytest

from specrag.ingest.fetch import DocumentRef, PdfRef, parse_attachments, parse_collection


class TestDocumentRefPart:
    @pytest.mark.parametrize(
        ("title", "expected"),
        [
            ("Fire safety: Approved Document B", "B"),
            ("Structure: Approved Document A", "A"),
            ("Protection from falling, collision and impact: Approved Document K", "K"),
            ("Approved Document B Volume 1: Dwellings", "B"),
        ],
    )
    def test_extracts_the_part_letter(self, title, expected):
        assert DocumentRef(title=title, base_path="/x").part == expected

    @pytest.mark.parametrize(
        "title",
        [
            "Manual to the Building Regulations",
            "The merged Approved Documents",
            "Building regulations and approved documents index",
        ],
    )
    def test_returns_none_for_collection_members_that_are_not_a_part(self, title):
        assert DocumentRef(title=title, base_path="/x").part is None

    def test_does_not_read_a_lowercase_word_as_a_part_letter(self):
        # Under a case-insensitive match this yields part "TO", because "to"
        # fits [A-Z]{1,2} and is followed by a word boundary.
        assert (
            DocumentRef(title="Approved Document to the regulations", base_path="/x").part is None
        )


class TestParseCollection:
    def test_reads_title_and_path_for_each_document(self):
        payload = {
            "links": {
                "documents": [
                    {"title": "Fire safety: Approved Document B", "base_path": "/pub/b"},
                    {"title": "Structure: Approved Document A", "base_path": "/pub/a"},
                ]
            }
        }
        refs = parse_collection(payload)
        assert [(r.title, r.base_path) for r in refs] == [
            ("Fire safety: Approved Document B", "/pub/b"),
            ("Structure: Approved Document A", "/pub/a"),
        ]

    def test_skips_entries_missing_a_title_or_path(self):
        payload = {
            "links": {
                "documents": [
                    {"title": "Fire safety: Approved Document B", "base_path": "/pub/b"},
                    {"title": "", "base_path": "/pub/x"},
                    {"title": "No path here"},
                ]
            }
        }
        assert len(parse_collection(payload)) == 1

    def test_returns_empty_for_a_payload_with_no_documents(self):
        assert parse_collection({}) == []
        assert parse_collection({"links": {}}) == []


class TestParseAttachments:
    def test_reads_pdf_attachments(self):
        payload = {
            "details": {
                "attachments": [
                    {
                        "title": "Approved Document K",
                        "url": "https://assets.example/AD_K.pdf",
                        "content_type": "application/pdf",
                        "file_size": 2112810,
                    }
                ]
            }
        }
        [pdf] = parse_attachments(payload, "K")
        assert pdf.part == "K"
        assert pdf.url == "https://assets.example/AD_K.pdf"
        assert pdf.size_bytes == 2112810

    def test_accepts_a_pdf_url_with_no_declared_content_type(self):
        payload = {"details": {"attachments": [{"title": "K", "url": "https://x/AD_K.PDF"}]}}
        assert len(parse_attachments(payload, "K")) == 1

    def test_skips_non_pdf_attachments(self):
        payload = {
            "details": {
                "attachments": [
                    {
                        "title": "Spreadsheet",
                        "url": "https://x/a.xlsx",
                        "content_type": "application/vnd.ms-excel",
                    },
                    {"title": "AD K", "url": "https://x/k.pdf", "content_type": "application/pdf"},
                ]
            }
        }
        [pdf] = parse_attachments(payload, "K")
        assert pdf.title == "AD K"

    def test_skips_attachments_with_no_url(self):
        payload = {
            "details": {"attachments": [{"title": "Broken", "content_type": "application/pdf"}]}
        }
        assert parse_attachments(payload, "K") == []

    def test_falls_back_to_a_generated_title(self):
        payload = {"details": {"attachments": [{"title": "  ", "url": "https://x/k.pdf"}]}}
        assert parse_attachments(payload, "K")[0].title == "Approved Document K"

    def test_returns_empty_when_the_publication_has_no_attachments(self):
        assert parse_attachments({}, "K") == []


class TestPdfRefFilename:
    def test_builds_a_stable_slugged_filename(self):
        ref = PdfRef(part="K", title="Approved Document K: protection from falling", url="u")
        assert ref.filename == "AD-K-approved-document-k-protection-from-falling.pdf"

    def test_handles_a_title_that_slugs_to_nothing(self):
        assert PdfRef(part="K", title="---", url="u").filename == "AD-K-document.pdf"
