"""Group cleaned page text into retrievable chunks.

Approved Documents are written as numbered clauses ("1.2 Measure the rise and
going..."), so chunks are built from clauses rather than a blind character
window. Keeping the clause number lets an answer cite a specific paragraph,
which is the whole point of retrieving over regulations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from specrag.config import MAX_CHUNK_CHARS, MIN_CHUNK_CHARS
from specrag.ingest.parse import Page

# A clause opens a line with its number and then real prose. The trailing
# [A-Z(] guard matters: without it a line such as "2.5 m clear width" in a
# dimensions table reads as clause 2.5.
_CLAUSE_RE = re.compile(r"^(\d+\.\d+)\s+(?=[A-Z(])")

_SENTENCE_END_RE = re.compile(r"(?<=[.;:])\s+")


@dataclass(frozen=True)
class Clause:
    """One numbered paragraph, with the section heading that introduced it."""

    number: str
    heading: str | None
    text: str
    page: int


@dataclass(frozen=True)
class Chunk:
    """A retrievable unit: one or more consecutive clauses."""

    chunk_id: str
    part: str
    heading: str | None
    clauses: tuple[str, ...]
    text: str
    page_start: int
    page_end: int

    @property
    def citation(self) -> str:
        if len(self.clauses) == 1:
            return f"Approved Document {self.part}, paragraph {self.clauses[0]}"
        return f"Approved Document {self.part}, paragraphs {self.clauses[0]}-{self.clauses[-1]}"

    def to_embedding_text(self) -> str:
        """Text handed to the embedding model.

        The heading is prepended because a clause often omits the subject its
        heading supplies: "1.2 Measure the rise and going" never says "stairs".
        """
        return f"{self.heading}\n{self.text}" if self.heading else self.text


def _looks_like_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 80:
        return False
    if _CLAUSE_RE.match(stripped) or stripped.endswith("."):
        return False
    return any(c.isalpha() for c in stripped)


def split_clauses(pages: list[Page], part: str) -> list[Clause]:
    """Walk the pages and emit one Clause per numbered paragraph.

    A line is treated as a section heading only when the next non-empty line
    opens a clause. Real headings introduce clauses; the loose label text that
    PDF extraction pulls out of diagrams ("Going", "Open riser", "Nosing") is
    followed by more labels, so this rule leaves it out of the heading slot.
    """
    lines: list[tuple[int, str]] = []
    for page in pages:
        for line in page.text.split("\n"):
            lines.append((page.number, line))

    clauses: list[Clause] = []
    heading: str | None = None
    current_number: str | None = None
    current_page = 0
    buffer: list[str] = []

    def flush() -> None:
        nonlocal buffer, current_number
        if current_number is not None:
            text = " ".join(part.strip() for part in buffer if part.strip()).strip()
            if text:
                clauses.append(
                    Clause(number=current_number, heading=heading, text=text, page=current_page)
                )
        buffer = []

    for index, (page_number, line) in enumerate(lines):
        match = _CLAUSE_RE.match(line.strip())
        if match:
            flush()
            current_number = match.group(1)
            current_page = page_number
            buffer = [line.strip()[match.end() :]]
            continue

        if _looks_like_heading(line):
            next_line = next(
                (text for _, text in lines[index + 1 :] if text.strip()),
                "",
            )
            if _CLAUSE_RE.match(next_line.strip()):
                flush()
                current_number = None
                heading = line.strip()
                continue

        if current_number is not None:
            buffer.append(line)

    flush()
    return clauses


def _split_oversized(text: str, limit: int) -> list[str]:
    """Break a clause longer than the limit on sentence boundaries."""
    if len(text) <= limit:
        return [text]
    pieces: list[str] = []
    current = ""
    for sentence in _SENTENCE_END_RE.split(text):
        candidate = f"{current} {sentence}".strip() if current else sentence
        if current and len(candidate) > limit:
            pieces.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def pack_chunks(
    clauses: list[Clause],
    part: str,
    max_chars: int = MAX_CHUNK_CHARS,
    min_chars: int = MIN_CHUNK_CHARS,
) -> list[Chunk]:
    """Merge short consecutive clauses and split long ones.

    Clauses are only merged while they share a heading, so a chunk never spans
    two sections of the document.
    """
    chunks: list[Chunk] = []
    pending: list[Clause] = []

    def emit(group: list[Clause]) -> None:
        if not group:
            return
        body = " ".join(f"{c.number} {c.text}" for c in group)
        numbers = tuple(c.number for c in group)
        span = numbers[0] if len(numbers) == 1 else f"{numbers[0]}-{numbers[-1]}"
        for piece_index, piece in enumerate(_split_oversized(body, max_chars)):
            suffix = f"#{piece_index + 1}" if piece_index else ""
            chunks.append(
                Chunk(
                    chunk_id=f"{part}:{span}{suffix}",
                    part=part,
                    heading=group[0].heading,
                    clauses=numbers,
                    text=piece,
                    page_start=min(c.page for c in group),
                    page_end=max(c.page for c in group),
                )
            )

    for clause in clauses:
        same_section = not pending or pending[0].heading == clause.heading
        projected = sum(len(c.text) for c in pending) + len(clause.text)
        if pending and (not same_section or projected > max_chars):
            emit(pending)
            pending = []
        pending.append(clause)
        if sum(len(c.text) for c in pending) >= min_chars:
            emit(pending)
            pending = []

    emit(pending)
    return chunks
