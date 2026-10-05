from __future__ import annotations

import pytest

from samadhan.evaluation.retrieval_metrics import ndcg_at_k, reciprocal_rank, score_ranking


def test_perfect_ranking() -> None:
    s = score_ranking(["A", "A", "B", "C"], ["A", "B"], k=5)
    assert s["hit@1"] == 1 and s["mrr"] == 1 and s["recall@5"] == 1 and s["ndcg@5"] == pytest.approx(1.0)


def test_chunks_are_collapsed_to_documents() -> None:
    # A,A counts as one document at rank 1; B is at rank 2 (not 3).
    assert score_ranking(["A", "A", "B"], ["B"], k=5)["mrr"] == pytest.approx(0.5)


def test_miss() -> None:
    s = score_ranking(["X", "Y"], ["A"], k=5)
    assert s["hit@5"] == 0 and s["mrr"] == 0 and s["ndcg@5"] == 0


def test_mrr_and_ndcg_values() -> None:
    assert reciprocal_rank(["X", "Y", "A"], {"A"}) == pytest.approx(1 / 3)
    # one relevant doc at rank 2 -> DCG = 1/log2(3); ideal = 1
    assert ndcg_at_k(["X", "A"], {"A"}, 5) == pytest.approx(0.6309, abs=1e-4)
