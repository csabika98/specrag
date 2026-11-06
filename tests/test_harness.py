"""Tests for the evaluation harness. Stub embedder, no model download."""

from __future__ import annotations

import json

import pytest

from specrag.eval.harness import evaluate, format_failures, load_questions
from specrag.index.embedder import HashingEmbedder
from specrag.index.store import VectorStore
from specrag.ingest.chunk import Chunk


@pytest.fixture
def embedder():
    return HashingEmbedder(dimension=256)


@pytest.fixture
def store(embedder):
    chunks = [
        Chunk("K:1.7", "K", "Nosings", ("1.7",),
              "Make step nosings apparent using a contrasting material a minimum of 55mm wide.", 1, 1),
        Chunk("K:1.34", "K", "Handrails", ("1.34",),
              "Position the top of the handrail 900mm to 1000mm from the pitch line or floor.", 2, 2),
        Chunk("K:2.8-2.9", "K", "Ramps", ("2.8", "2.9"),
              "Design ramps with a maximum slope of 1:12 and a minimum width of 1500mm.", 3, 3),
        Chunk("K:5.4", "K", "Glazing", ("5.4",),
              "Glazing in a critical location should satisfy Class 3 of BS EN 12600.", 4, 4),
    ]
    return VectorStore.build(chunks, embedder)


def write_questions(tmp_path, rows):
    path = tmp_path / "questions.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


class TestLoadQuestions:
    def test_reads_every_field(self, tmp_path):
        path = write_questions(tmp_path, [{
            "id": "q1", "question": "How high?", "relevant_clauses": ["1.34"],
            "expected": "900mm to 1000mm.", "part": "K",
        }])
        [question] = load_questions(path)
        assert question.id == "q1"
        assert question.relevant_clauses == frozenset({"1.34"})
        assert question.part == "K"

    def test_skips_blank_lines(self, tmp_path):
        path = tmp_path / "q.jsonl"
        row = {"id": "q1", "question": "x", "relevant_clauses": ["1.1"], "expected": "y", "part": "K"}
        path.write_text(json.dumps(row) + "\n\n\n", encoding="utf-8")
        assert len(load_questions(path)) == 1

    def test_the_shipped_golden_set_loads_and_has_unique_ids(self):
        questions = load_questions()
        assert len(questions) >= 10
        assert len({q.id for q in questions}) == len(questions)
        assert all(q.relevant_clauses for q in questions)


class TestEvaluate:
    def test_finds_a_clause_that_is_present(self, store, embedder, tmp_path):
        path = write_questions(tmp_path, [{
            "id": "q", "question": "What width must a contrasting step nosing be?",
            "relevant_clauses": ["1.7"], "expected": "55mm", "part": "K",
        }])
        report, [result] = evaluate(store, embedder, load_questions(path))
        assert result.found_at is not None
        assert report.hit_at_5 == 1.0

    def test_records_a_miss_when_no_clause_matches(self, store, embedder, tmp_path):
        path = write_questions(tmp_path, [{
            "id": "q", "question": "anything", "relevant_clauses": ["9.9"],
            "expected": "nothing", "part": "K",
        }])
        report, [result] = evaluate(store, embedder, load_questions(path))
        assert result.found_at is None
        assert report.hit_at_5 == 0.0

    def test_a_merged_chunk_satisfies_either_of_its_clauses(self, store, embedder, tmp_path):
        path = write_questions(tmp_path, [{
            "id": "q", "question": "What is the maximum ramp slope?",
            "relevant_clauses": ["2.9"], "expected": "1:12", "part": "K",
        }])
        _, [result] = evaluate(store, embedder, load_questions(path))
        assert result.found_at is not None  # chunk K:2.8-2.9 carries 2.9

    def test_respects_the_retrieval_depth(self, store, embedder, tmp_path):
        path = write_questions(tmp_path, [{
            "id": "q", "question": "glazing class critical location",
            "relevant_clauses": ["5.4"], "expected": "Class 3", "part": "K",
        }])
        _, [result] = evaluate(store, embedder, load_questions(path), k=1)
        assert len(result.retrieved) == 1

    def test_an_empty_question_set_produces_an_empty_report(self, store, embedder):
        report, results = evaluate(store, embedder, [])
        assert report.questions == 0 and results == []


class TestFormatFailures:
    def test_reports_no_misses(self, store, embedder, tmp_path):
        path = write_questions(tmp_path, [{
            "id": "q", "question": "step nosing contrasting width",
            "relevant_clauses": ["1.7"], "expected": "55mm", "part": "K",
        }])
        _, results = evaluate(store, embedder, load_questions(path))
        assert format_failures(results) == "No misses."

    def test_names_the_missed_question_and_what_was_wanted(self, store, embedder, tmp_path):
        path = write_questions(tmp_path, [{
            "id": "missing-one", "question": "unrelated", "relevant_clauses": ["9.9"],
            "expected": "nothing", "part": "K",
        }])
        _, results = evaluate(store, embedder, load_questions(path))
        text = format_failures(results)
        assert "missing-one" in text and "9.9" in text
