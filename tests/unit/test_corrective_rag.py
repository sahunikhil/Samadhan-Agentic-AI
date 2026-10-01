"""Corrective-RAG control flow with a stub retriever: correct / incorrect->retry / ambiguous paths."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from caseflow.config import Settings
from caseflow.llm import ModelRegistry
from caseflow.rag.graph import GroundedAnswer, SearchPlan, build_knowledge_graph
from caseflow.rag.stores.base import SearchHit
from tests.fakes import ScriptedChatModel, text_of


class StubRetriever:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.calls: list[dict[str, Any]] = []

    async def retrieve(self, query: str, **kwargs: Any) -> list[SearchHit]:
        self.calls.append({"query": query, **kwargs})
        return [
            SearchHit(
                chunk_id=f"c{i}",
                doc_id=f"KB-{i}",
                title="T",
                url="",
                section="S",
                category="billing",
                text=f"chunk {i}",
                score=1.0,
                rerank_score=s,
            )
            for i, s in enumerate(self.scores)
        ]


def _graph(scores: list[float], answerable: bool = True) -> tuple[Any, StubRetriever, ScriptedChatModel]:
    def responder(messages: list[BaseMessage], tools: list[str], schema: type[BaseModel] | None) -> Any:
        if schema is SearchPlan:
            return SearchPlan(query="rewritten query", category="billing")
        if schema is GroundedAnswer:
            n_excerpts = text_of(messages).count("> S\n")
            return GroundedAnswer(answer=f"answer from {n_excerpts} excerpts [1]", cited=[1], answerable=answerable)
        raise AssertionError("unexpected call")

    fake = ScriptedChatModel(responder=responder)
    models = ModelRegistry(Settings().llm)
    models.override("fast", fake)
    models.override("smart", fake)
    stub = StubRetriever(scores)
    return build_knowledge_graph(Settings(), models, stub), stub, fake  # type: ignore[arg-type]


async def test_correct_path_uses_only_chunks_above_threshold() -> None:
    graph, stub, _ = _graph([0.99, 0.6, 0.01])
    out = await graph.ainvoke({"question": "Do gift cards expire?"})
    assert out["answerable"] and "2 excerpts" in out["answer"]
    assert len(stub.calls) == 1
    # Recall with rewrite + original question; precision judged against the original question.
    assert stub.calls[0]["extra_queries"] == ["Do gift cards expire?"]
    assert stub.calls[0]["rerank_query"] == "Do gift cards expire?"


async def test_low_scores_retry_once_then_take_the_ambiguous_path() -> None:
    graph, stub, _ = _graph([0.02, 0.01, 0.01, 0.0])
    out = await graph.ainvoke({"question": "Do you offer financing?"})
    assert len(stub.calls) == 2, "incorrect -> transform query -> retrieve again"
    assert stub.calls[1]["category"] is None, "the retry drops the category filter"
    assert "3 excerpts" in out["answer"], "ambiguous: top-3 chunks go to the grounded generator"


@pytest.mark.parametrize("answerable", [False])
async def test_generator_still_abstains_when_excerpts_do_not_answer(answerable: bool) -> None:
    graph, _, _ = _graph([0.01, 0.01], answerable=answerable)
    out = await graph.ainvoke({"question": "Do you sell a smart fridge?"})
    assert out["answerable"] is False
