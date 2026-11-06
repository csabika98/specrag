"""Tests for retrieval metrics."""

from __future__ import annotations

import pytest

from specrag.eval.metrics import aggregate, hit_at_k, recall_at_k, reciprocal_rank


def ranking(*clause_groups: str) -> list[set[str]]:
    """Build a ranking from space-separated clause groups, best first."""
    return [set(group.split()) for group in clause_groups]


class TestHitAtK:
    def test_true_when_the_top_result_is_relevant(self):
        assert hit_at_k(ranking("1.7", "2.1"), {"1.7"}, 1)

    def test_false_when_the_relevant_clause_is_below_k(self):
        assert not hit_at_k(ranking("2.1", "3.1", "1.7"), {"1.7"}, 2)

    def test_true_when_the_relevant_clause_is_at_position_k(self):
        assert hit_at_k(ranking("2.1", "3.1", "1.7"), {"1.7"}, 3)

    def test_true_when_a_merged_chunk_carries_the_clause(self):
        # Chunks merge consecutive clauses, so membership is a set overlap.
        assert hit_at_k(ranking("1.6 1.7 1.8"), {"1.7"}, 1)

    def test_false_when_nothing_relevant_was_retrieved(self):
        assert not hit_at_k(ranking("2.1", "3.1"), {"1.7"}, 5)

    def test_false_on_an_empty_ranking(self):
        assert not hit_at_k([], {"1.7"}, 5)


class TestReciprocalRank:
    @pytest.mark.parametrize(
        ("rank_position", "expected"), [(1, 1.0), (2, 0.5), (3, 1 / 3), (4, 0.25)]
    )
    def test_is_the_inverse_of_the_first_relevant_position(self, rank_position, expected):
        groups = ["9.9"] * (rank_position - 1) + ["1.7"]
        assert reciprocal_rank(ranking(*groups), {"1.7"}) == pytest.approx(expected)

    def test_is_zero_when_nothing_relevant_is_found(self):
        assert reciprocal_rank(ranking("2.1", "3.1"), {"1.7"}) == 0.0

    def test_respects_a_cutoff(self):
        assert reciprocal_rank(ranking("2.1", "1.7"), {"1.7"}, k=1) == 0.0

    def test_is_zero_on_an_empty_ranking(self):
        assert reciprocal_rank([], {"1.7"}) == 0.0


class TestRecallAtK:
    def test_counts_the_share_of_relevant_clauses_found(self):
        assert recall_at_k(ranking("1.7", "2.1"), {"1.7", "1.8"}, 5) == 0.5

    def test_is_one_when_every_relevant_clause_appears(self):
        assert recall_at_k(ranking("1.7", "1.8"), {"1.7", "1.8"}, 5) == 1.0

    def test_ignores_clauses_below_the_cutoff(self):
        assert recall_at_k(ranking("1.7", "1.8"), {"1.7", "1.8"}, 1) == 0.5

    def test_a_merged_chunk_can_satisfy_several_relevant_clauses(self):
        assert recall_at_k(ranking("1.7 1.8"), {"1.7", "1.8"}, 1) == 1.0

    def test_is_zero_when_nothing_is_relevant(self):
        assert recall_at_k(ranking("2.1"), set(), 5) == 0.0


class TestAggregate:
    def test_averages_across_questions(self):
        report = aggregate([
            (ranking("1.7"), {"1.7"}),          # hit@1
            (ranking("9.9", "2.1"), {"2.1"}),   # hit@3 but not hit@1
        ])
        assert report.questions == 2
        assert report.hit_at_1 == 0.5
        assert report.hit_at_3 == 1.0
        assert report.mrr == pytest.approx(0.75)

    def test_an_empty_run_scores_zero_rather_than_dividing_by_zero(self):
        report = aggregate([])
        assert report.questions == 0 and report.mrr == 0.0

    def test_table_renders_every_metric(self):
        table = aggregate([(ranking("1.7"), {"1.7"})]).as_table()
        for label in ("questions", "hit@1", "hit@3", "hit@5", "MRR", "recall@5"):
            assert label in table
