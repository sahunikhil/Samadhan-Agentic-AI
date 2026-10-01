"""Knowledge specialist: a Corrective-RAG (CRAG) LangGraph subgraph.

    START -> cache_lookup --(verified hit)--------------------------------------------> END
                  |
                  v
             plan_search -> retrieve -> grade --(relevant)--> generate -> cache_write -> END
                  ^                        |
                  +----(nothing relevant, retry once with a broader query)

Design decisions
----------------
* **Workflow, not agent.** Retrieval-QA follows a predictable path, so an explicit
  graph is cheaper (<= 2 LLM calls), faster and easier to evaluate than letting a
  model loop over a search tool. Agents are reserved for the order/returns work
  where the next step genuinely depends on tool results.
* **Cheap retrieval evaluator.** CRAG (Yan et al., 2024) grades retrieved documents
  with a small evaluator model, not a big LLM. We use the cross-encoder we already
  run for reranking: chunks above ``relevance_threshold`` count as relevant. Zero
  extra LLM calls, deterministic, and it's the signal that decides "retry vs answer".
* **Query transformation on failure.** A failed search is retried once with a
  broader, filter-free rewrite before we admit we don't know.
* **Grounded, cited generation** with a structured output (``answerable`` flag) so
  "I don't know" is an explicit, measurable outcome rather than a hallucination.
* **Node-level caching** on ``retrieve`` (keyed by query + filter): repeated FAQ
  questions skip embedding + search + rerank entirely.
* **Semantic answer cache** (optional, ``rag/semantic_cache.py``): similar questions are
  candidates, an LLM verifier confirms equivalence, only confident answers are stored.
* ``input_schema`` / ``output_schema`` keep the subgraph's public contract small
  while it uses richer internal state.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from langchain_core.exceptions import OutputParserException
from langgraph.cache.memory import InMemoryCache
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import CachePolicy, Command, RetryPolicy, default_retry_on
from pydantic import BaseModel, Field

from caseflow.config import Settings
from caseflow.llm import ModelRegistry
from caseflow.prompts import GROUNDED_ANSWER, SEARCH_PLANNER, SEARCH_RETRY_INSTRUCTIONS
from caseflow.rag.retriever import HybridRetriever
from caseflow.rag.semantic_cache import SemanticCache

KBCategory = Literal[
    "shipping", "returns", "warranty", "billing", "orders", "account", "support", "product", "troubleshooting"
]
MAX_SEARCH_ATTEMPTS = 2
AMBIGUOUS_TOP_N = 3


class SearchPlan(BaseModel):
    """A standalone help-center search query."""

    query: str = Field(description="Keyword-rich standalone search query")
    category: KBCategory | None = Field(default=None, description="Best category filter, or null")


class GroundedAnswer(BaseModel):
    """An answer grounded in the provided excerpts."""

    answer: str = Field(description="Answer with inline [n] citations")
    cited: list[int] = Field(default_factory=list, description="Excerpt numbers actually used")
    answerable: bool = Field(description="False if the excerpts don't contain the answer")


class KnowledgeInput(TypedDict, total=False):
    question: str
    context: str


class KnowledgeOutput(TypedDict, total=False):
    answer: str
    answerable: bool
    citations: list[dict[str, Any]]
    contexts: list[str]
    search_queries: list[str]
    cached: bool


class KnowledgeState(KnowledgeInput, KnowledgeOutput, total=False):
    search_query: str
    category: str | None
    attempts: int
    documents: list[dict[str, Any]]
    relevant: list[dict[str, Any]]
    grade_action: Literal["correct", "incorrect", "ambiguous"]


def _retry_on(exc: Exception) -> bool:
    # Retry malformed structured outputs (common on small free models) + default transient errors.
    return isinstance(exc, OutputParserException) or default_retry_on(exc)


def _retrieve_cache_key(state: KnowledgeState) -> str:
    return f"{state.get('question', '')}|{state.get('search_query', '')}|{state.get('category') or '*'}"


def build_knowledge_graph(
    settings: Settings, models: ModelRegistry, retriever: HybridRetriever, cache: SemanticCache | None = None
) -> CompiledStateGraph:
    company = settings.agent.company_name
    threshold = settings.retrieval.relevance_threshold

    async def plan_search(state: KnowledgeState) -> dict[str, Any]:
        attempts = state.get("attempts", 0)
        retry = SEARCH_RETRY_INSTRUCTIONS.format(previous_query=state.get("search_query", "")) if attempts > 0 else ""
        planner = models.structured("fast", SearchPlan)
        plan: SearchPlan = await planner.ainvoke(  # type: ignore[assignment]
            SEARCH_PLANNER.format(
                company=company,
                question=state["question"],
                context=state.get("context") or "(none)",
                retry_instructions=retry,
            )
        )
        return {
            "search_query": plan.query,
            "category": None if attempts > 0 else plan.category,
            "attempts": attempts + 1,
            "search_queries": [*state.get("search_queries", []), plan.query],
        }

    async def retrieve(state: KnowledgeState) -> dict[str, Any]:
        # Multi-query recall (rewrite + original question), precision judged against the ORIGINAL
        # question: rewrites help find candidates but can mislead the reranker.
        question, category = state["question"], state.get("category")
        kwargs: dict[str, Any] = {"extra_queries": [question], "rerank_query": question}
        hits = await retriever.retrieve(state["search_query"], category=category, **kwargs)
        if not hits and category:
            # A wrong category guess should not starve retrieval: fall back to the whole KB.
            hits = await retriever.retrieve(state["search_query"], **kwargs)
        return {"documents": [h.model_dump() for h in hits]}

    async def grade(state: KnowledgeState) -> Command[Literal["generate", "plan_search"]]:
        """CRAG's three actions, with the cross-encoder as the retrieval evaluator.

        * correct   - some chunks clear the threshold -> answer from those.
        * incorrect - nothing clears it and we can retry -> transform the query, search again.
        * ambiguous - still nothing after the retry -> hand the top chunks to the generator and let
          its grounded ``answerable`` flag decide. An absolute threshold on an uncalibrated
          cross-encoder produced false abstentions in live evals; the generator is told to abstain
          if the excerpts don't contain the answer, so this doesn't invite hallucination.
        """
        docs = state.get("documents", [])
        # Without a reranker there is no calibrated score: trust the retriever's top-k.
        relevant = [d for d in docs if d.get("rerank_score") is None or d["rerank_score"] >= threshold]
        if relevant:
            return Command(update={"relevant": relevant, "grade_action": "correct"}, goto="generate")
        if state.get("attempts", 0) < MAX_SEARCH_ATTEMPTS:
            return Command(update={"relevant": [], "grade_action": "incorrect"}, goto="plan_search")
        return Command(update={"relevant": docs[:AMBIGUOUS_TOP_N], "grade_action": "ambiguous"}, goto="generate")

    async def generate(state: KnowledgeState) -> dict[str, Any]:
        relevant = state.get("relevant", [])
        if not relevant:
            return {
                "answer": "The help center does not cover this question.",
                "answerable": False,
                "citations": [],
                "contexts": [],
            }
        blocks = "\n\n".join(
            f"[{i}] {d['title']} > {d['section']}\n{d['text']}" for i, d in enumerate(relevant, start=1)
        )
        writer = models.structured("smart", GroundedAnswer)
        result: GroundedAnswer = await writer.ainvoke(  # type: ignore[assignment]
            GROUNDED_ANSWER.format(company=company, documents=blocks, question=state["question"])
        )
        cited = [n for n in result.cited if 1 <= n <= len(relevant)] or list(range(1, len(relevant) + 1))
        citations = [
            {
                "n": n,
                "doc_id": relevant[n - 1]["doc_id"],
                "title": relevant[n - 1]["title"],
                "url": relevant[n - 1]["url"],
            }
            for n in cited
        ]
        return {
            "answer": result.answer,
            "answerable": result.answerable,
            "citations": citations,
            # Exactly what the generator saw (header + text). Evaluating faithfulness against bare
            # chunk text made correct answers look unsupported: claims that rely on the header
            # ("...on the VoltBook") had no evidence in the judge's context.
            "contexts": [f"{d['title']} > {d['section']}\n{d['text']}" for d in relevant],
        }

    async def cache_lookup(state: KnowledgeState) -> Command[Literal["plan_search", "__end__"]]:
        assert cache is not None
        hit = await cache.lookup(state["question"])
        if hit is None:
            return Command(goto="plan_search")
        v = hit.value
        return Command(
            update={
                "answer": v["answer"],
                "answerable": True,
                "citations": v.get("citations", []),
                "contexts": v.get("contexts", []),
                "search_queries": [],
                "cached": True,
            },
            goto="__end__",  # END; the literal keeps the Command type precise
        )

    async def cache_write(state: KnowledgeState) -> dict[str, Any]:
        # Only confident answers: grounded, answerable, and found on the "correct" CRAG path.
        if cache is not None and state.get("answerable") and state.get("grade_action") == "correct":
            await cache.put(
                state["question"],
                {k: state.get(k) for k in ("answer", "citations", "contexts")},
            )
        return {"cached": False}

    llm_retry = RetryPolicy(max_attempts=2, initial_interval=1.0, retry_on=_retry_on)
    builder = StateGraph(KnowledgeState, input_schema=KnowledgeInput, output_schema=KnowledgeOutput)
    builder.add_node("plan_search", plan_search, retry_policy=llm_retry)
    builder.add_node("retrieve", retrieve, cache_policy=CachePolicy(key_func=_retrieve_cache_key, ttl=600))
    builder.add_node("grade", grade, destinations=("generate", "plan_search"))
    builder.add_node("generate", generate, retry_policy=llm_retry)
    builder.add_edge("plan_search", "retrieve")
    builder.add_edge("retrieve", "grade")
    if cache is not None:
        builder.add_node("cache_lookup", cache_lookup, destinations=("plan_search", END))
        builder.add_node("cache_write", cache_write)
        builder.add_edge(START, "cache_lookup")
        builder.add_edge("generate", "cache_write")
        builder.add_edge("cache_write", END)
    else:
        builder.add_edge(START, "plan_search")
        builder.add_edge("generate", END)
    return builder.compile(name="knowledge_agent", cache=InMemoryCache())
