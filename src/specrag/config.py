"""Paths and model defaults. Everything configurable lives here."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.environ.get("SPECRAG_DATA", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PARSED_DIR = DATA_DIR / "parsed"
INDEX_DIR = DATA_DIR / "index"
GOLDEN_DIR = DATA_DIR / "golden"

# Small enough to embed a full corpus on a CPU-only machine with ~1GB free RAM.
EMBEDDING_MODEL = os.environ.get("SPECRAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_BATCH_SIZE = int(os.environ.get("SPECRAG_EMBEDDING_BATCH", "16"))

# Chunking. Approved Documents are written as numbered clauses, so chunks are
# built from clauses rather than a blind character window.
MAX_CHUNK_CHARS = int(os.environ.get("SPECRAG_MAX_CHUNK_CHARS", "1200"))
MIN_CHUNK_CHARS = int(os.environ.get("SPECRAG_MIN_CHUNK_CHARS", "200"))

GOVUK_API = "https://www.gov.uk/api/content"
APPROVED_DOCUMENTS_COLLECTION = "/government/collections/approved-documents"
