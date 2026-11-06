"""Discover and download the Approved Documents from gov.uk.

The gov.uk Content API is used rather than scraping HTML: it is a stable,
documented interface, and it gives us the attachment URLs directly.

Content is Crown copyright, published under the Open Government Licence v3.0.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import httpx

from specrag.config import APPROVED_DOCUMENTS_COLLECTION, GOVUK_API, RAW_DIR

# "Fire safety: Approved Document B" -> part "B".
#
# The part letter must be uppercase in the source. Matching case-insensitively
# here would read "Approved Document to the ..." as part "TO", because "to"
# satisfies [A-Z]{1,2} under IGNORECASE and is followed by a word boundary.
_PART_RE = re.compile(r"[Aa]pproved [Dd]ocument\s+([A-Z]{1,2})\b")


@dataclass(frozen=True)
class DocumentRef:
    """A publication page in the Approved Documents collection."""

    title: str
    base_path: str

    @property
    def part(self) -> str | None:
        """The regulation part letter, e.g. 'B' for fire safety.

        Returns None for collection members that are not a lettered part, such
        as the Manual to the Building Regulations or the merged index page.
        """
        match = _PART_RE.search(self.title)
        return match.group(1) if match else None


@dataclass(frozen=True)
class PdfRef:
    """A downloadable PDF attachment belonging to a part."""

    part: str
    title: str
    url: str
    size_bytes: int | None = None

    @property
    def filename(self) -> str:
        return f"AD-{self.part}-{_slug(self.title)}.pdf"


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:60] or "document"


def parse_collection(payload: dict) -> list[DocumentRef]:
    """Extract publication references from a collection API response.

    Pure function over the API payload so it can be tested without network.
    """
    documents = payload.get("links", {}).get("documents", [])
    refs = []
    for doc in documents:
        title = (doc.get("title") or "").strip()
        base_path = doc.get("base_path") or ""
        if title and base_path:
            refs.append(DocumentRef(title=title, base_path=base_path))
    return refs


def parse_attachments(payload: dict, part: str) -> list[PdfRef]:
    """Extract PDF attachments from a publication API response.

    Non-PDF attachments (ordering pages, spreadsheets) are skipped.
    """
    attachments = payload.get("details", {}).get("attachments", [])
    pdfs = []
    for att in attachments:
        url = att.get("url") or ""
        content_type = att.get("content_type") or ""
        is_pdf = content_type == "application/pdf" or url.lower().endswith(".pdf")
        if not is_pdf or not url:
            continue
        pdfs.append(
            PdfRef(
                part=part,
                title=(att.get("title") or "").strip() or f"Approved Document {part}",
                url=url,
                size_bytes=att.get("file_size"),
            )
        )
    return pdfs


def discover(client: httpx.Client | None = None) -> list[PdfRef]:
    """Walk the collection and return every lettered part's PDF attachments."""
    owns_client = client is None
    client = client or httpx.Client(timeout=30.0, follow_redirects=True)
    try:
        collection = client.get(f"{GOVUK_API}{APPROVED_DOCUMENTS_COLLECTION}").json()
        pdfs: list[PdfRef] = []
        for ref in parse_collection(collection):
            part = ref.part
            if part is None:
                continue
            publication = client.get(f"{GOVUK_API}{ref.base_path}").json()
            pdfs.extend(parse_attachments(publication, part))
        return pdfs
    finally:
        if owns_client:
            client.close()


def download(pdfs: list[PdfRef], dest: Path | None = None) -> list[Path]:
    """Download each PDF, skipping any already present at the expected size."""
    dest = dest or RAW_DIR
    dest.mkdir(parents=True, exist_ok=True)
    paths = []
    with httpx.Client(timeout=120.0, follow_redirects=True) as client:
        for pdf in pdfs:
            target = dest / pdf.filename
            if target.exists() and (
                pdf.size_bytes is None or target.stat().st_size == pdf.size_bytes
            ):
                paths.append(target)
                continue
            response = client.get(pdf.url)
            response.raise_for_status()
            target.write_bytes(response.content)
            paths.append(target)
    return paths
