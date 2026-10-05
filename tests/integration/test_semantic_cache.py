"""Semantic cache: exact / verified / rejected hits, safety rules, invalidation, and the CRAG graph wiring.

Uses the real bge-small embeddings (local ONNX) and a scripted verifier.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableLambda
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel

from samadhan.config import Settings
from samadhan.llm import ModelRegistry
from samadhan.rag.embeddings import FastEmbedEmbeddings
from samadhan.rag.graph import GroundedAnswer, SearchPlan, build_knowledge_graph
from samadhan.rag.semantic_cache import CacheVerdict, SemanticCache, is_cacheable, kb_version_from_hashes
from tests.fakes import ScriptedChatModel
from tests.unit.test_corrective_rag import StubRetriever

ANSWER = {
    "answer": "Opened earbuds are returnable only with the hygiene seal intact [1].",
    "citations": [],
    "contexts": [],
}


class Verifier:
    """Scripted equivalence verifier that records its calls."""

    def __init__(self, same: bool) -> None:
        self.same = same
        self.calls = 0

    def runnable(self) -> Any:
        async def run(_: str) -> CacheVerdict:
            self.calls += 1
            return CacheVerdict(same_answer=self.same)

        return RunnableLambda(run)


def _cache(embeddings: Any, verifier: Verifier | None, version: list[str] | None = None, **kw: Any) -> SemanticCache:
    store = InMemoryStore(index={"dims": embeddings.dense_dim, "embed": FastEmbedEmbeddings(embeddings)})
    version = version or ["v1"]

    async def kb_version() -> str:
        return version[0]

    return SemanticCache(
        store,
        kb_version=kb_version,
        verifier=verifier.runnable() if verifier else None,
        company="Voltwise",
        version_refresh_s=0,
        **kw,
    )


def test_personal_questions_are_never_cached() -> None:
    assert is_cacheable("Can I return opened earbuds?")
    for q in ("Where is VW-10003?", "refund to jane@example.com", "card 4242 4242", "where is my order?", " "):
        assert not is_cacheable(q), q
    assert kb_version_from_hashes({"a": "1", "b": "2"}) == kb_version_from_hashes({"b": "2", "a": "1"})


async def test_exact_repeat_hits_without_calling_the_verifier(embeddings: Any) -> None:
    verifier = Verifier(same=False)
    cache = _cache(embeddings, verifier)
    assert await cache.put("Can I return opened earbuds?", ANSWER)
    hit = await cache.lookup("can I return OPENED earbuds")
    assert hit is not None and not hit.verified and hit.value["answer"] == ANSWER["answer"]
    assert verifier.calls == 0


async def test_paraphrase_needs_the_verifiers_yes(embeddings: Any) -> None:
    yes, no = Verifier(same=True), Verifier(same=False)
    for verifier, expect_hit in ((yes, True), (no, False)):
        cache = _cache(embeddings, verifier)
        await cache.put("Can I return opened earbuds?", ANSWER)
        hit = await cache.lookup("Are opened earbuds returnable?")
        assert (hit is not None) is expect_hit and verifier.calls == 1


async def test_without_a_verifier_only_exact_matches_are_served(embeddings: Any) -> None:
    cache = _cache(embeddings, None)
    await cache.put("Can I return opened earbuds?", ANSWER)
    assert await cache.lookup("Are opened earbuds returnable?") is None
    assert await cache.lookup("Can I return opened earbuds?") is not None


async def test_unrelated_questions_never_reach_the_verifier(embeddings: Any) -> None:
    verifier = Verifier(same=True)
    cache = _cache(embeddings, verifier)
    await cache.put("Can I return opened earbuds?", ANSWER)
    assert await cache.lookup("How do I reset my password?") is None
    assert verifier.calls == 0  # below the candidate threshold: no LLM cost for obvious misses


async def test_kb_change_and_ttl_invalidate(embeddings: Any) -> None:
    version = ["v1"]
    cache = _cache(embeddings, Verifier(same=True), version)
    await cache.put("Can I return opened earbuds?", ANSWER)
    version[0] = "v2"  # an article was re-ingested -> new namespace
    assert await cache.lookup("Can I return opened earbuds?") is None

    expired = _cache(embeddings, Verifier(same=True), ttl_minutes=0)
    await expired.put("Can I return opened earbuds?", ANSWER)
    assert await expired.lookup("Can I return opened earbuds?") is None


async def test_knowledge_graph_serves_repeat_questions_from_the_cache(embeddings: Any) -> None:
    llm_calls: list[str] = []

    def responder(messages: list[BaseMessage], tools: list[str], schema: type[BaseModel] | None) -> Any:
        llm_calls.append(schema.__name__ if schema else "text")
        if schema is SearchPlan:
            return SearchPlan(query="opened earbuds return", category="returns")
        if schema is GroundedAnswer:
            return GroundedAnswer(answer="Only with the hygiene seal intact [1].", cited=[1], answerable=True)
        raise AssertionError(schema)

    models = ModelRegistry(Settings().llm)
    models.override("fast", ScriptedChatModel(responder=responder))
    models.override("smart", ScriptedChatModel(responder=responder))
    cache = _cache(embeddings, Verifier(same=True))
    graph = build_knowledge_graph(Settings(), models, StubRetriever([0.99]), cache=cache)  # type: ignore[arg-type]

    first = await graph.ainvoke({"question": "Can I return opened earbuds?"})
    assert not first.get("cached") and llm_calls == ["SearchPlan", "GroundedAnswer"]

    second = await graph.ainvoke({"question": "Are opened earbuds returnable?"})
    assert second["cached"] and second["answer"] == first["answer"]
    assert llm_calls == ["SearchPlan", "GroundedAnswer"], "no RAG LLM calls on a cache hit"

    # Personal questions bypass the cache entirely.
    await graph.ainvoke({"question": "Can I return opened earbuds from order VW-10003?"})
    assert len(llm_calls) == 4
