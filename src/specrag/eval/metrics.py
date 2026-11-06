"""Retrieval metrics.

All of these take the retrieved ranking as a list of clause sets, one per
chunk, in rank order. A chunk counts as a hit when it carries any clause that
answers the question, because a chunk may merge several consecutive clauses.

No LLM is involved, which is deliberate: retrieval quality is measurable on
its own, it is the half of a RAG system that fails quietly, and keeping it
model-free means CI can run the whole evaluation on every pull request.
"""

from __future__ import annotations

from dataclasses import dataclass


def hit_at_k(retrieved: list[set[str]], relevant: set[str], k: int) -> bool:
    """True when any of the top k chunks carries a relevant clause."""
    return any(chunk & relevant for chunk in retrieved[:k])


def reciprocal_rank(retrieved: list[set[str]], relevant: set[str], k: int | None = None) -> float:
    """1/rank of the first chunk carrying a relevant clause, else 0.0."""
    for position, chunk in enumerate(retrieved if k is None else retrieved[:k], start=1):
        if chunk & relevant:
            return 1.0 / position
    return 0.0


def recall_at_k(retrieved: list[set[str]], relevant: set[str], k: int) -> float:
    """Share of the relevant clauses that appear anywhere in the top k."""
    if not relevant:
        return 0.0
    found = set().union(*retrieved[:k]) if retrieved[:k] else set()
    return len(found & relevant) / len(relevant)


@dataclass(frozen=True)
class Report:
    """Aggregate scores over a question set."""

    questions: int
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    mrr: float
    recall_at_5: float

    def as_table(self) -> str:
        rows = [
            ("questions", f"{self.questions}"),
            ("hit@1", f"{self.hit_at_1:.3f}"),
            ("hit@3", f"{self.hit_at_3:.3f}"),
            ("hit@5", f"{self.hit_at_5:.3f}"),
            ("MRR", f"{self.mrr:.3f}"),
            ("recall@5", f"{self.recall_at_5:.3f}"),
        ]
        width = max(len(name) for name, _ in rows)
        return "\n".join(f"{name:<{width}}  {value}" for name, value in rows)


def aggregate(results: list[tuple[list[set[str]], set[str]]]) -> Report:
    """Average the metrics over every (ranking, relevant set) pair."""
    if not results:
        return Report(0, 0.0, 0.0, 0.0, 0.0, 0.0)
    count = len(results)
    return Report(
        questions=count,
        hit_at_1=sum(hit_at_k(r, g, 1) for r, g in results) / count,
        hit_at_3=sum(hit_at_k(r, g, 3) for r, g in results) / count,
        hit_at_5=sum(hit_at_k(r, g, 5) for r, g in results) / count,
        mrr=sum(reciprocal_rank(r, g) for r, g in results) / count,
        recall_at_5=sum(recall_at_k(r, g, 5) for r, g in results) / count,
    )
