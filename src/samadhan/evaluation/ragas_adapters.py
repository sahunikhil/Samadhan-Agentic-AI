"""Glue between Samadhan and RAGAS 0.4.

RAGAS 0.4 "collections" metrics need two things:

* an **instructor-based LLM** (``ragas.llms.llm_factory``). Every free provider we use
  (Gemini, Groq, Ollama) exposes an *OpenAI-compatible* endpoint, so one code path
  covers all of them: ``AsyncOpenAI(base_url=..., api_key=...)`` + ``provider="openai"``.
  (RAGAS itself recommends the OpenAI-compatible route for Gemini because of an
  upstream instructor issue with the native Google SDK.)
* a **BaseRagasEmbedding** - we wrap our local FastEmbed model, so answer-relevancy is
  computed with the same free embedding model the retriever uses.

We also convert LangChain messages to RAGAS messages ourselves: the built-in
``ragas.integrations.langgraph.convert_to_ragas_messages`` reads OpenAI-specific
``additional_kwargs`` and rejects list-typed content (which Gemini returns), so it
silently loses tool calls for non-OpenAI models.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from ragas.embeddings.base import BaseRagasEmbedding

from samadhan.config import LLMSettings
from samadhan.rag.embeddings import EmbeddingModels

OPENAI_COMPATIBLE: dict[str, tuple[str, tuple[str, ...]]] = {
    "google_genai": ("https://generativelanguage.googleapis.com/v1beta/openai/", ("GOOGLE_API_KEY", "GEMINI_API_KEY")),
    "groq": ("https://api.groq.com/openai/v1", ("GROQ_API_KEY",)),
    "openai": ("https://api.openai.com/v1", ("OPENAI_API_KEY",)),
}


def judge_llm(settings: LLMSettings) -> Any:
    """RAGAS judge LLM for the configured ``judge`` role, via an OpenAI-compatible endpoint."""
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory

    spec = settings.model_for("judge")
    provider, _, model = spec.partition(":")
    if provider == "ollama":
        client = AsyncOpenAI(base_url=f"{settings.ollama_base_url.rstrip('/')}/v1", api_key="ollama")
    else:
        if provider not in OPENAI_COMPATIBLE:
            raise ValueError(f"No OpenAI-compatible endpoint known for judge provider '{provider}'")
        base_url, env_names = OPENAI_COMPATIBLE[provider]
        api_key = next((os.environ[n] for n in env_names if os.environ.get(n)), None)
        if not api_key:
            raise RuntimeError(f"Judge model {spec} needs one of {env_names} in the environment")
        client = AsyncOpenAI(base_url=base_url, api_key=api_key, max_retries=4, timeout=120)
    kwargs: dict[str, Any] = {"temperature": 0.0, "max_tokens": 4096}
    if provider == "groq" and model.startswith("qwen/qwen3"):
        # Groq's free tier caps this model at 1,000 *output tokens per minute* and rejects larger
        # requests outright ("Request too large", no retry helps); a thinking model spends most of
        # that on reasoning. Claim extraction / verification don't need chain-of-thought: disable
        # thinking and stay under the cap. (Found live: judge calls failed and silently produced
        # faithfulness 0.08 and factual correctness 0.0 for correct, cited answers.)
        kwargs |= {"reasoning_effort": "none", "max_tokens": 1000}
    return llm_factory(model, provider="openai", client=client, **kwargs)


def batched_faithfulness(judge: Any, batch_size: int = 5) -> Any:
    """RAGAS ``Faithfulness`` hardened for small output budgets.

    Stock behavior verifies *all* statements in one judge call and scores
    ``supported / len(returned verdicts)``. With a capped output budget (Groq free tier: 1,000 output
    tokens/minute for the judge) a 13-statement answer's verdict JSON is cut off - a parse failure at
    best, a score computed from a partial list at worst. Here: verdicts in batches of ``batch_size``,
    and a result is refused (NaN upstream) unless every statement received exactly one verdict.
    """
    from ragas.metrics.collections import Faithfulness
    from ragas.metrics.collections.faithfulness.util import NLIStatementOutput

    class BatchedFaithfulness(Faithfulness):  # type: ignore[misc]
        async def _create_verdicts(self, statements: list[str], context: str) -> Any:
            verdicts: list[Any] = []
            for i in range(0, len(statements), batch_size):
                batch = statements[i : i + batch_size]
                out = await Faithfulness._create_verdicts(self, batch, context)
                if len(out.statements) != len(batch):
                    raise ValueError(f"judge returned {len(out.statements)} verdicts for {len(batch)} statements")
                verdicts.extend(out.statements)
            return NLIStatementOutput(statements=verdicts)

    return BatchedFaithfulness(llm=judge)


class FastEmbedRagasEmbedding(BaseRagasEmbedding):
    def __init__(self, models: EmbeddingModels) -> None:
        super().__init__()
        self._models = models

    def embed_text(self, text: str, **kwargs: Any) -> list[float]:
        return next(iter(self._models._get_dense().query_embed(text))).tolist()  # type: ignore[no-any-return]

    async def aembed_text(self, text: str, **kwargs: Any) -> list[float]:
        return await asyncio.to_thread(self.embed_text, text)


def to_ragas_messages(messages: list[BaseMessage]) -> list[Any]:
    """LangChain -> RAGAS messages, provider-agnostic (reads ``.tool_calls`` / ``.text``)."""
    import ragas.messages as r

    out: list[Any] = []
    for m in messages:
        if isinstance(m, HumanMessage):
            out.append(r.HumanMessage(content=m.text))
        elif isinstance(m, AIMessage):
            calls = [r.ToolCall(name=tc["name"], args=tc.get("args", {})) for tc in m.tool_calls] or None
            out.append(r.AIMessage(content=m.text, tool_calls=calls))
        elif isinstance(m, ToolMessage):
            out.append(r.ToolMessage(content=m.text))
    return out
