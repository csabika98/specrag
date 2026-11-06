"""Run the golden question set against an index and report retrieval scores."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from specrag.config import GOLDEN_DIR
from specrag.eval.metrics import Report, aggregate
from specrag.index.embedder import Embedder
from specrag.index.store import VectorStore


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    relevant_clauses: frozenset[str]
    expected: str
    part: str


@dataclass(frozen=True)
class QuestionResult:
    question: Question
    retrieved: list[set[str]]
    top_citation: str | None
    found_at: int | None  # 1-based rank of the first relevant chunk


def load_questions(path: Path | None = None) -> list[Question]:
    path = path or (GOLDEN_DIR / "questions.jsonl")
    questions = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            questions.append(
                Question(
                    id=row["id"],
                    question=row["question"],
                    relevant_clauses=frozenset(row["relevant_clauses"]),
                    expected=row["expected"],
                    part=row["part"],
                )
            )
    return questions


def evaluate(
    store: VectorStore,
    embedder: Embedder,
    questions: list[Question],
    k: int = 5,
) -> tuple[Report, list[QuestionResult]]:
    """Score retrieval over the question set.

    No generation happens here. Retrieval is the half of a RAG system that
    fails silently -- a fluent answer built on the wrong clause still reads
    well -- so it is measured on its own.
    """
    results: list[QuestionResult] = []
    for question in questions:
        hits = store.search(question.question, embedder, k=k)
        retrieved = [set(chunk.clauses) for chunk, _ in hits]
        found_at = next(
            (i for i, clauses in enumerate(retrieved, start=1) if clauses & question.relevant_clauses),
            None,
        )
        results.append(
            QuestionResult(
                question=question,
                retrieved=retrieved,
                top_citation=hits[0][0].citation if hits else None,
                found_at=found_at,
            )
        )
    report = aggregate([(r.retrieved, set(r.question.relevant_clauses)) for r in results])
    return report, results


def format_failures(results: list[QuestionResult], k: int = 5) -> str:
    """List the questions whose answer never appeared, for error analysis."""
    misses = [r for r in results if r.found_at is None]
    if not misses:
        return "No misses."
    lines = [f"{len(misses)} question(s) with no relevant clause in the top {k}:"]
    for result in misses:
        wanted = ", ".join(sorted(result.question.relevant_clauses))
        lines.append(f"  [{result.question.id}] wanted {wanted}; top hit was {result.top_citation}")
        lines.append(f"      {result.question.question}")
    return "\n".join(lines)
