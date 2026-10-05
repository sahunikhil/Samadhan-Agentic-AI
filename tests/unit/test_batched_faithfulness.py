"""Faithfulness hardened for small judge output budgets: batched verdicts, no score from partial lists."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("ragas")

from ragas.metrics.collections import Faithfulness
from ragas.metrics.collections.faithfulness.util import NLIStatementOutput, StatementFaithfulnessAnswer

from samadhan.evaluation.ragas_adapters import batched_faithfulness

STATEMENTS = [f"claim {i}" for i in range(13)]


def _patch(monkeypatch: pytest.MonkeyPatch, *, drop_last: bool = False) -> list[int]:
    batch_sizes: list[int] = []

    async def statements(self: Any, question: str, response: str) -> list[str]:
        return STATEMENTS

    async def verdicts(self: Any, batch: list[str], context: str) -> NLIStatementOutput:
        batch_sizes.append(len(batch))
        items = [StatementFaithfulnessAnswer(statement=s, reason="in context", verdict=1) for s in batch]
        return NLIStatementOutput(statements=items[:-1] if drop_last else items)

    monkeypatch.setattr(Faithfulness, "_create_statements", statements)
    monkeypatch.setattr(Faithfulness, "_create_verdicts", verdicts)
    return batch_sizes


def _metric() -> Any:
    # A real RAGAS LLM wrapper around a dummy client: constructing it makes no network call, and the
    # patched statement/verdict steps never reach it.
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory

    judge = llm_factory("dummy", provider="openai", client=AsyncOpenAI(api_key="test", base_url="http://127.0.0.1:9"))
    return batched_faithfulness(judge, batch_size=5)


async def test_verdicts_are_requested_in_small_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    sizes = _patch(monkeypatch)
    result = await _metric().ascore(user_input="q", response="a", retrieved_contexts=["ctx"])
    assert sizes == [5, 5, 3] and result.value == 1.0


async def test_a_truncated_verdict_list_is_an_error_not_a_score(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, drop_last=True)
    with pytest.raises(ValueError, match="verdicts for"):
        await _metric().ascore(user_input="q", response="a", retrieved_contexts=["ctx"])
