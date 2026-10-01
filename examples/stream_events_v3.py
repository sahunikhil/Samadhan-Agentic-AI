"""LangGraph 1.2 event streaming (``stream_events(version="v3")``) on the Corrective-RAG graph.

The production API streams with the stable ``astream(version="v2")``; v3 is
marked experimental in LangGraph 1.2. This script shows what v3 adds: one ordered stream of
typed *protocol events* on named channels (``lifecycle``, ``updates``, ``messages``, ``tools``,
``custom`` ...), each with a namespace path and a monotonically increasing ``seq``.

    uv run python examples/stream_events_v3.py "Can I bring the PowerCell 20K on a plane?"

Needs an LLM key (see .env.example). Uses an in-memory vector index, so no servers are needed.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

from caseflow.config import get_settings
from caseflow.llm import ModelRegistry
from caseflow.rag.embeddings import EmbeddingModels
from caseflow.rag.graph import build_knowledge_graph
from caseflow.rag.ingest import ingest_knowledge_base
from caseflow.rag.retriever import HybridRetriever
from caseflow.rag.stores.qdrant import QdrantHybridStore


class EventPrinter:
    """Render protocol events compactly: token deltas are buffered per message and printed on
    ``message-finish``, split into *reasoning* and *text* content blocks (v3 keeps them apart)."""

    def __init__(self) -> None:
        self.text: list[str] = []
        self.reasoning: list[str] = []

    def __call__(self, event: dict[str, Any]) -> None:
        method, params = event["method"], event["params"]
        ns = "/".join(params.get("namespace") or []) or "root"
        data = params.get("data")
        seq = f"[{event['seq']:>4}]"
        if method == "lifecycle" and isinstance(data, dict):
            print(f"{seq} lifecycle {ns}: {data.get('event')} {data.get('graph_name') or ''}")
        elif method == "updates" and isinstance(data, dict):
            print(f"{seq} updates   {ns}: {', '.join(data)} finished")
        elif method == "messages":
            payload = data[0] if isinstance(data, list | tuple) and data else data
            if not isinstance(payload, dict):
                return
            kind = payload.get("event")
            if kind == "message-start":
                self.text, self.reasoning = [], []
                print(f"{seq} messages  {ns}: LLM call started")
            elif kind == "content-block-delta":
                delta = payload.get("delta") or {}
                if delta.get("type") == "reasoning-delta":
                    self.reasoning.append(delta.get("reasoning", ""))
                else:
                    self.text.append(delta.get("text", ""))
            elif kind == "message-finish":
                if self.reasoning:
                    print(f"{seq} reasoning {ns}: {''.join(self.reasoning)[:160]!r}...")
                if self.text:
                    print(f"{seq} text      {ns}: {''.join(self.text)[:200]!r}")


async def main(question: str) -> None:
    settings = get_settings()
    embeddings = EmbeddingModels(settings.retrieval)
    await asyncio.to_thread(embeddings.warmup)
    store = QdrantHybridStore(collection="example", path=Path(":memory:"))
    await ingest_knowledge_base(settings, store, embeddings, force=True)
    graph = build_knowledge_graph(
        settings, ModelRegistry(settings.llm), HybridRetriever(store, embeddings, settings.retrieval)
    )

    printer = EventPrinter()
    stream = await graph.astream_events({"question": question}, version="v3")
    async for event in stream:  # raw ProtocolEvents, strictly ordered by `seq`
        printer(event)  # type: ignore[arg-type]
    output = await stream.output()
    print("\nanswerable:", (output or {}).get("answerable"))
    print("answer:", (output or {}).get("answer"))
    print("citations:", [c["title"] for c in (output or {}).get("citations", [])])
    await store.close()


if __name__ == "__main__":
    asyncio.run(main(" ".join(sys.argv[1:]) or "Can I bring the PowerCell 20K power bank on a plane?"))
