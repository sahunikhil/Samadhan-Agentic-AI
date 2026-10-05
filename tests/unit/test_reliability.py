"""pass@k / pass^k estimators (tau-bench style reliability)."""

from __future__ import annotations

import pytest

from samadhan.evaluation.reliability import pass_at_k, pass_hat_k, reliability_summary


def test_estimators_match_their_definitions() -> None:
    # 3 of 4 trials succeed.
    assert pass_hat_k(4, 3, 1) == pytest.approx(0.75)
    assert pass_hat_k(4, 3, 2) == pytest.approx(0.5)  # C(3,2)/C(4,2) = 3/6
    assert pass_hat_k(4, 3, 4) == 0.0  # a single failure means "not reliable at k=n"
    assert pass_at_k(4, 3, 2) == pytest.approx(1.0)  # any 2 trials include a success
    assert pass_at_k(4, 1, 2) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        pass_hat_k(2, 3, 1)


def test_reliability_drops_with_k_for_inconsistent_agents() -> None:
    summary = reliability_summary({"S1": [True, True, True], "S2": [True, False, True], "S3": [False] * 3})
    assert summary["pass^1"] == pytest.approx((1 + 2 / 3 + 0) / 3, abs=1e-4)
    assert summary["pass^3"] == pytest.approx(1 / 3, abs=1e-4)  # only S1 is reliable
    assert summary["pass^1"] > summary["pass^2"] > summary["pass^3"]
    assert summary["pass@3"] == pytest.approx(2 / 3, abs=1e-4)
