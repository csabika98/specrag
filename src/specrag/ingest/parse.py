"""Turn an Approved Document PDF into clean, page-numbered text.

The published PDFs carry three kinds of noise that wreck retrieval if left in:

* an ``ONLINE VERSION`` watermark, stamped roughly four times per page
* a running header naming the document and edition, alternating between
  recto and verso layout
* typographic ligatures (``ﬁ``, ``ﬂ``, ``ﬃ``) that break literal matching,
  so "profiles" in a query never meets "proﬁles" in the text

Everything here is a pure function over text except :func:`read_pdf`.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

WATERMARK = "ONLINE VERSION"

# The running header names the part and edition, e.g.
# "Approved Document K, 2013 edition". It cannot be matched on that full
# phrase, because the PDF content stream splits the header mid-word and
# extraction yields fragments such as "t K, 2013 edition", sometimes doubled.
# The stable signal is "<part letter>, <year> edition".
_HEADER_FRAGMENT_RE = re.compile(r"[A-Za-z]*\s*[A-Z]{1,2},\s*\d{4}\s+edition", re.IGNORECASE)

# Residue that legitimately surrounds a header fragment on the same line.
_HEADER_RESIDUE_RE = re.compile(
    r"(Building\s*Regulations|Approved|Documen\w*|\d{1,4}|[\s\-|*])",
    re.IGNORECASE,
)

# A line that is only a page number.
_PAGE_NUMBER_RE = re.compile(r"^\s*\d{1,4}\s*$")


@dataclass(frozen=True)
class Page:
    number: int
    text: str


def normalise(text: str) -> str:
    """Fold ligatures and unify whitespace, leaving line structure intact.

    NFKC turns ``ﬁ`` into ``fi``; without it a search for "fire office" cannot
    match text the typesetter rendered with ligatures.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("–", "-").replace("—", "-")
    text = text.replace("\xa0", " ")
    # Collapse runs of spaces/tabs but keep newlines, which carry clause structure.
    text = re.sub(r"[ \t]+", " ", text)
    return text


def strip_watermark(text: str) -> str:
    """Remove the watermark, including a copy truncated by the page boundary."""
    text = text.replace(WATERMARK, "")
    # Extraction often ends mid-watermark, e.g. "ONLINE VER".
    for cut in range(len(WATERMARK) - 1, 2, -1):
        suffix = WATERMARK[:cut]
        if text.rstrip().endswith(suffix):
            text = text.rstrip()[: -len(suffix)]
            break
    return text


def is_running_header(line: str) -> bool:
    """True for a running header or a bare page number.

    A line counts as a header only when removing the header fragments and the
    boilerplate around them leaves essentially nothing. That distinction
    matters: the cover carries "2013 edition - for use in England" and the
    contents list "Main changes in the 2013 edition", both of which mention an
    edition and are body text. Neither contains the "<part>, <year> edition"
    fragment, and both leave substantial residue.
    """
    stripped = line.strip()
    if not stripped:
        return False
    if _PAGE_NUMBER_RE.match(stripped):
        return True
    if not _HEADER_FRAGMENT_RE.search(stripped):
        return False
    residue = _HEADER_FRAGMENT_RE.sub("", stripped)
    residue = _HEADER_RESIDUE_RE.sub("", residue)
    return len(residue) < 4


def clean_page(text: str) -> str:
    """Normalise a page and drop watermark and running-header lines."""
    text = strip_watermark(normalise(text))
    kept = [line.rstrip() for line in text.split("\n") if not is_running_header(line)]
    # Drop blank lines left behind, but keep single separators between blocks.
    out: list[str] = []
    for line in kept:
        if line.strip() or (out and out[-1].strip()):
            out.append(line)
    return "\n".join(out).strip()


def read_pdf(path: Path) -> list[Page]:
    """Extract cleaned text per page.

    The gov.uk PDFs are AES-encrypted with an empty owner password, so pypdf
    needs the ``cryptography`` extra present to open them at all.
    """
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = []
    for index, page in enumerate(reader.pages, start=1):
        pages.append(Page(number=index, text=clean_page(page.extract_text() or "")))
    return pages
