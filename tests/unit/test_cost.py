"""Cost accounting: price lookup, cached-input discount, per-turn tracker, unknown models."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from caseflow.cost import ModelPrice, TurnCostTracker, call_cost, price_for


def test_price_lookup_handles_provider_prefixes_overrides_and_versions() -> None:
    assert price_for("groq:openai/gpt-oss-120b") == price_for("openai/gpt-oss-120b")
    assert price_for("gemini-3.5-flash-lite-001") == price_for("gemini-3.5-flash-lite")  # not "gemini-3.5-flash"
    assert price_for("qwen3:8b") is None  # local model: not in the table
    override = {"qwen3:8b": ModelPrice(input=0, output=0)}
    assert price_for("qwen3:8b", override) == ModelPrice(input=0, output=0)


def test_call_cost_applies_the_cached_input_discount() -> None:
    price = ModelPrice(input=0.15, output=0.60, cached_input=0.075)
    assert call_cost(price, 10_000, 1_000) == pytest.approx(0.0021)
    # 8K of the 10K input tokens were a provider prompt-cache hit.
    assert call_cost(price, 10_000, 1_000, cached_input_tokens=8_000) == pytest.approx(0.0015)


async def test_tracker_sums_every_call_of_a_run() -> None:
    tracker = TurnCostTracker()
    for model, tokens_in, tokens_out in (
        ("openai/gpt-oss-20b", 2_000, 200),
        ("openai/gpt-oss-120b", 5_000, 500),
        ("unknown-local-model", 1_000, 100),
    ):
        msg = AIMessage(
            content="x",
            usage_metadata={
                "input_tokens": tokens_in,
                "output_tokens": tokens_out,
                "total_tokens": tokens_in + tokens_out,
            },
            response_metadata={"model_name": model},
        )
        await tracker.on_llm_end(LLMResult(generations=[[ChatGeneration(message=msg)]]))
    usage = tracker.summary()
    assert usage["llm_calls"] == 3 and usage["input_tokens"] == 8_000 and usage["output_tokens"] == 800
    expected = (2_000 * 0.075 + 200 * 0.30 + 5_000 * 0.15 + 500 * 0.60) / 1e6
    assert usage["cost_usd"] == pytest.approx(expected)
    assert usage["by_model"]["unknown-local-model"]["cost_usd"] == 0  # counted as unpriced, not guessed
