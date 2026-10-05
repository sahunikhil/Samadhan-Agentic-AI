"""Typed loaders for the golden evaluation datasets in ``evals/datasets``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

DATASETS = Path("evals/datasets")


class RetrievalCase(BaseModel):
    id: str
    question: str
    relevant: list[str]


class RAGCase(BaseModel):
    id: str
    question: str
    reference: str
    reference_doc_ids: list[str] = Field(default_factory=list)

    @property
    def answerable(self) -> bool:
        return bool(self.reference_doc_ids)


class ToolCallSpec(BaseModel):
    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class AgentScenario(BaseModel):
    id: str
    customer_id: str
    message: str
    expected_agents: list[str]
    expected_outcome: str
    reference_tool_calls: list[ToolCallSpec] = Field(default_factory=list)
    forbidden_tools: list[str] = Field(default_factory=list)
    reference_goal: str
    on_topic: bool = True
    resolve: dict[str, Any] = Field(default_factory=dict)


class RedTeamCase(BaseModel):
    id: str
    customer_id: str
    category: str
    attack: str
    expect_blocked: bool
    must_not_contain: list[str] = Field(default_factory=list)


class CachePair(BaseModel):
    """A cached question and a new question; ``same`` = one answer serves both."""

    id: str
    cached: str
    query: str
    same: bool
    kind: str  # paraphrase | near_miss | unrelated
    slot: str | None = None  # what differs in a near miss (product, condition, aspect ...)


def _load(name: str, model: type[BaseModel], base: Path = DATASETS) -> list[Any]:
    path = base / name
    return [
        model.model_validate(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def retrieval_cases(base: Path = DATASETS) -> list[RetrievalCase]:
    return _load("retrieval_golden.jsonl", RetrievalCase, base)


def rag_cases(base: Path = DATASETS) -> list[RAGCase]:
    return _load("rag_golden.jsonl", RAGCase, base)


def agent_scenarios(base: Path = DATASETS) -> list[AgentScenario]:
    return _load("agent_scenarios.jsonl", AgentScenario, base)


def redteam_cases(base: Path = DATASETS) -> list[RedTeamCase]:
    return _load("redteam.jsonl", RedTeamCase, base)


def cache_pairs(base: Path = DATASETS) -> list[CachePair]:
    return _load("semantic_cache_pairs.jsonl", CachePair, base)
