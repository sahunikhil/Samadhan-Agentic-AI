"""LLM cost accounting: price table, per-turn cost tracking, budgets.

Why this matters: tokens, not servers, dominate the bill of an LLM product, and cost per resolved
conversation is a first-class product metric next to resolution rate and latency. We track it at
three levels:

* **Per turn** - :class:`TurnCostTracker` is a LangChain callback attached to each graph run. It
  sees every model call (triage, specialists, CRAG, synthesizer, guard, memory) with the model's
  own ``usage_metadata`` and returns ``{input_tokens, output_tokens, cached_input_tokens,
  cost_usd, by_model}`` in the final SSE event and the eval rows.
* **Fleet** - ``caseflow_llm_cost_usd_total{model}`` (Prometheus) for dashboards and alerts.
* **Budget** - ``agent.turn_budget_usd``: a turn that exceeds it is logged and counted
  (``caseflow_turn_over_budget_total``). The *hard* limits are the per-turn model/tool call limits
  in the agent middleware, which bound cost deterministically; a budget breach is a signal to
  investigate (loops, huge tool results, prompt bloat), not something to cut off mid-answer.

Prices are configuration, not code: override with ``CASEFLOW_OBSERVABILITY__PRICES`` (JSON), e.g.
``{"openai/gpt-oss-120b": {"input": 0.15, "output": 0.60, "cached_input": 0.075}}``.
Free tiers cost $0 - we report *list-price equivalent* spend so capacity planning is honest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import LLMResult
from pydantic import BaseModel

from caseflow.observability import LLM_COST, LLM_UNPRICED, get_logger

log = get_logger(__name__)


class ModelPrice(BaseModel):
    """USD per 1M tokens."""

    input: float
    output: float
    cached_input: float | None = None


# List prices (USD / 1M tokens), checked Sept 2026: GroqDocs model pages; Google AI pricing page
# (Gemini 3.x Flash family). Local models (Ollama) cost $0 in tokens - you pay for the GPU instead.
DEFAULT_PRICES: dict[str, ModelPrice] = {
    "openai/gpt-oss-120b": ModelPrice(input=0.15, output=0.60, cached_input=0.075),
    "openai/gpt-oss-20b": ModelPrice(input=0.075, output=0.30, cached_input=0.0375),
    "qwen/qwen3.8-27b": ModelPrice(input=0.80, output=4.00),
    "gemini-3.5-flash": ModelPrice(input=1.50, output=9.00),
    "gemini-3.5-flash-lite": ModelPrice(input=0.30, output=2.50),
}


def _key(model: str) -> str:
    return model.split(":", 1)[1] if model.split(":", 1)[0] in {"groq", "google_genai", "ollama", "openai"} else model


def price_for(model: str, prices: dict[str, ModelPrice] | None = None) -> ModelPrice | None:
    table = {**DEFAULT_PRICES, **(prices or {})}
    key = _key(model)
    if key in table:
        return table[key]
    # Provider responses sometimes add a version suffix (e.g. "...-001"): fall back to a prefix match.
    return next((p for name, p in sorted(table.items(), key=lambda kv: -len(kv[0])) if key.startswith(name)), None)


def call_cost(price: ModelPrice, input_tokens: int, output_tokens: int, cached_input_tokens: int = 0) -> float:
    cached = min(cached_input_tokens, input_tokens)
    cached_price = price.cached_input if price.cached_input is not None else price.input
    return ((input_tokens - cached) * price.input + cached * cached_price + output_tokens * price.output) / 1_000_000


@dataclass
class ModelUsage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    cost_usd: float = 0.0


class TurnCostTracker(AsyncCallbackHandler):
    """Accumulates usage and list-price cost for one graph run (all nested LLM calls)."""

    def __init__(self, prices: dict[str, ModelPrice] | None = None) -> None:
        super().__init__()
        self.prices = prices or {}
        self.by_model: dict[str, ModelUsage] = {}

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        for generations in response.generations:
            for gen in generations:
                message = getattr(gen, "message", None)
                usage = getattr(message, "usage_metadata", None) if message is not None else None
                if not usage:
                    continue
                meta = getattr(message, "response_metadata", {}) or {}
                model = str(meta.get("model_name") or meta.get("model") or "unknown")
                self.record(
                    model,
                    int(usage.get("input_tokens", 0)),
                    int(usage.get("output_tokens", 0)),
                    int((usage.get("input_token_details") or {}).get("cache_read", 0)),
                )

    def record(self, model: str, input_tokens: int, output_tokens: int, cached_input_tokens: int = 0) -> None:
        u = self.by_model.setdefault(model, ModelUsage())
        u.calls += 1
        u.input_tokens += input_tokens
        u.output_tokens += output_tokens
        u.cached_input_tokens += cached_input_tokens
        price = price_for(model, self.prices)
        if price is None:
            LLM_UNPRICED.labels(model=model).inc()
            return
        cost = call_cost(price, input_tokens, output_tokens, cached_input_tokens)
        u.cost_usd += cost
        LLM_COST.labels(model=model).inc(cost)

    def summary(self) -> dict[str, Any]:
        return {
            "llm_calls": sum(u.calls for u in self.by_model.values()),
            "input_tokens": sum(u.input_tokens for u in self.by_model.values()),
            "output_tokens": sum(u.output_tokens for u in self.by_model.values()),
            "cached_input_tokens": sum(u.cached_input_tokens for u in self.by_model.values()),
            "cost_usd": round(sum(u.cost_usd for u in self.by_model.values()), 6),
            "by_model": {m: {**u.__dict__, "cost_usd": round(u.cost_usd, 6)} for m, u in sorted(self.by_model.items())},
        }
