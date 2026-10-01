"""Semantic answer cache for the knowledge specialist - with an equivalence verifier.

Why cache?
    Help-center questions repeat ("can I return opened earbuds?" arrives hundreds of times a day
    in different words). A full Corrective-RAG answer costs 2 LLM calls (~4-8K tokens), retrieval
    and reranking (~0.4 s) and several seconds of generation. A cache hit costs one embedding
    lookup and, usually, one verifier call (measured: ~470 tokens incl. reasoning on gpt-oss-20b).

Why not a plain similarity threshold?  (measured: ``caseflow eval cache``)
    Embeddings encode *topic*, not *equivalence*. On our labeled pairs, bge-small scores
    "Can I cancel an order **after** it ships?" vs "... **before** it ships?" at **0.96** and
    "opened earbuds" vs "opened headphones" (different return rules!) at 0.95 - higher than most
    true paraphrases. The threshold that avoids every wrong answer (0.96) serves only 18% of
    paraphrases; at 0.80, embeddings alone would serve a wrong answer to 46% of near misses. The reranker cross-encoder fails the same way: it is trained for query-passage
    *relevance*, and a near-miss question is highly relevant.

The design: retrieve, then verify (like RAG itself)
    1. **Candidate** - Store semantic search, similarity >= ``cache_candidate_threshold``
       (tuned for recall: it only decides what is worth verifying).
    2. **Exact** - identical normalized text -> hit, no LLM call.
    3. **Verify** - the fast model answers one structured question: "is an answer to the cached
       question a complete, correct answer to the new one?" (strict: product, condition, aspect
       must match; when unsure -> no). Precision comes from here.

    Measured (48 labeled pairs, gpt-oss-20b verifier): paraphrase hit rate 0.86, false-hit rate
    0.00, precision 1.00 - gated in CI (``caseflow eval cache``).

Safety rules (a cache is shared across customers)
    * Only knowledge answers are cached - never order, refund or account data.
    * Questions that look personal (order ids, emails, card-like numbers, "my order") are neither
      looked up nor stored.
    * Only confident answers are stored: ``answerable`` and CRAG action ``correct``.
    * The namespace includes a **KB version** (hash of every indexed document's content hash) and
      the **prompt version**: re-ingesting a changed article or shipping a new prompt invalidates
      every entry automatically, across replicas, without a flush.
    * TTL bounds staleness (native Store TTL on Postgres; ``created_at`` check elsewhere).
    * Any cache error fails open to the normal RAG path.

Storage is the LangGraph ``BaseStore`` the app already runs (InMemory in dev, Postgres+pgvector
in production), so the cache is shared by every replica with no extra infrastructure.
"""

from __future__ import annotations

import hashlib
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.runnables import Runnable
from langgraph.store.base import BaseStore
from pydantic import BaseModel, Field

from caseflow.observability import SEMANTIC_CACHE, get_logger
from caseflow.prompts import CACHE_EQUIVALENCE, PROMPT_VERSION

log = get_logger(__name__)

NAMESPACE = "semantic_cache"
_PERSONAL = re.compile(
    r"\bVW-\d+|\bRF-\d+|\bRT-\d+|[\w.+-]+@[\w-]+\.[\w.]+|\d{4,}|\bmy (order|refund|return|account|package|parcel)\b",
    re.IGNORECASE,
)

CacheResult = Literal["hit_exact", "hit_verified", "rejected", "miss", "skipped", "error"]


class CacheVerdict(BaseModel):
    """Whether a cached answer can be reused for a new question."""

    same_answer: bool = Field(description="True only if an answer to the cached question fully answers the new one")


@dataclass(frozen=True, slots=True)
class CacheHit:
    value: dict[str, Any]
    similarity: float
    cached_question: str
    verified: bool


def normalize(question: str) -> str:
    return " ".join(re.sub(r"[^\w\s+]", " ", question.lower()).split())


def is_cacheable(question: str) -> bool:
    """Shared-cache safety: nothing customer-specific goes in or comes out."""
    return bool(question.strip()) and _PERSONAL.search(question) is None and len(question) <= 500


def kb_version_from_hashes(hashes: dict[str, str]) -> str:
    return hashlib.sha256("|".join(f"{k}:{v}" for k, v in sorted(hashes.items())).encode()).hexdigest()[:12]


class SemanticCache:
    def __init__(
        self,
        store: BaseStore,
        *,
        kb_version: Callable[[], Awaitable[str]],
        verifier: Runnable[Any, Any] | None,
        company: str,
        candidate_threshold: float = 0.80,
        ttl_minutes: int = 1440,
        version_refresh_s: float = 60.0,
    ) -> None:
        self.store = store
        self._kb_version_fn = kb_version
        self._verifier = verifier
        self._company = company
        self.candidate_threshold = candidate_threshold
        self.ttl_minutes = ttl_minutes
        self._version_refresh_s = version_refresh_s
        self._version: tuple[str, float] | None = None

    async def namespace(self) -> tuple[str, ...]:
        # The KB version is re-read at most once a minute: a re-ingest on any replica invalidates
        # every replica's view within ``version_refresh_s``. Store labels may not contain '.'.
        now = time.monotonic()
        if self._version is None or now - self._version[1] > self._version_refresh_s:
            self._version = (await self._kb_version_fn(), now)
        return (NAMESPACE, self._version[0], PROMPT_VERSION.replace(".", "_"))

    async def lookup(self, question: str) -> CacheHit | None:
        result: CacheResult = "miss"
        try:
            hit, result = await self._lookup(question)
            return hit
        except Exception:  # a cache must never break answering
            log.exception("semantic_cache_lookup_failed")
            result = "error"
            return None
        finally:
            SEMANTIC_CACHE.labels(result=result).inc()

    async def _lookup(self, question: str) -> tuple[CacheHit | None, CacheResult]:
        if not is_cacheable(question):
            return None, "skipped"
        items = await self.store.asearch(await self.namespace(), query=question, limit=3)
        oldest = time.time() - self.ttl_minutes * 60
        candidates = [
            i
            for i in items
            if i.score is not None and i.score >= self.candidate_threshold and i.value.get("created_at", 0) >= oldest
        ]
        if not candidates:
            return None, "miss"
        norm = normalize(question)
        for item in candidates:
            if normalize(item.value["question"]) == norm:
                return CacheHit(item.value, float(item.score or 0), item.value["question"], verified=False), "hit_exact"
        if self._verifier is None:  # without a verifier, only exact matches are safe
            return None, "rejected"
        best = candidates[0]
        verdict: CacheVerdict = await self._verifier.ainvoke(
            CACHE_EQUIVALENCE.format(company=self._company, cached=best.value["question"], new=question)
        )
        log.info("semantic_cache_verify", similarity=round(best.score or 0, 3), same=verdict.same_answer)
        if not verdict.same_answer:
            return None, "rejected"
        return CacheHit(best.value, float(best.score or 0), best.value["question"], verified=True), "hit_verified"

    async def put(self, question: str, value: dict[str, Any]) -> bool:
        if not is_cacheable(question):
            return False
        try:
            kwargs: dict[str, Any] = {"ttl": self.ttl_minutes} if self.store.supports_ttl else {}
            await self.store.aput(
                await self.namespace(),
                hashlib.sha256(normalize(question).encode()).hexdigest()[:32],
                {**value, "question": question, "created_at": time.time()},
                index=["question"],
                **kwargs,
            )
            return True
        except Exception:
            log.exception("semantic_cache_put_failed")
            return False


def store_has_vector_index(store: Any) -> bool:
    """Semantic lookup needs a Store configured with an embedding index."""
    return getattr(store, "index_config", None) is not None
