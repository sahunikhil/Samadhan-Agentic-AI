"""Semantic-cache evaluation: is a cache hit safe to serve?

For every labeled pair (``evals/datasets/semantic_cache_pairs.jsonl``) we compute the similarity
exactly as the production path does (Store: ``embed_documents`` for the cached question,
``embed_query`` for the new one, cosine) and then:

1. **Threshold sweep** (no LLM) - for each candidate threshold: paraphrase hit rate (recall),
   false-hit rate on near misses/unrelated pairs, precision. This shows *why* a similarity
   threshold alone is unsafe and picks the candidate threshold (high recall is what matters there).
2. **Verified** (``--verify``, fast LLM) - candidates at the configured threshold go through the
   equivalence verifier; the final false-hit rate is gated at 0.

Cost/benefit is reported too: a verifier call measured ~470 tokens against ~4-8K for a full RAG answer.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from samadhan.config import Settings
from samadhan.evaluation.datasets import CachePair, cache_pairs
from samadhan.evaluation.report import SuiteResult, mean
from samadhan.llm import ModelRegistry
from samadhan.observability import TokenUsageCallback
from samadhan.prompts import CACHE_EQUIVALENCE, PROMPT_VERSION
from samadhan.rag.embeddings import EmbeddingModels, FastEmbedEmbeddings
from samadhan.rag.semantic_cache import CacheVerdict, normalize

THRESHOLDS = (0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.97)


def _cos(a: list[float], b: list[float]) -> float:
    va, vb = np.asarray(a), np.asarray(b)
    return float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb)))


def _rates(pairs: list[CachePair], hits: list[bool]) -> dict[str, float]:
    same = [h for p, h in zip(pairs, hits, strict=True) if p.same]
    diff = [h for p, h in zip(pairs, hits, strict=True) if not p.same]
    tp, fp = sum(same), sum(diff)
    return {
        "paraphrase_hit_rate": round(tp / len(same), 4) if same else float("nan"),
        "false_hit_rate": round(fp / len(diff), 4) if diff else float("nan"),
        "precision": round(tp / (tp + fp), 4) if tp + fp else float("nan"),
    }


def _previous_verdicts(sink: Path | None) -> dict[str, dict[str, Any]]:
    if sink is None or not sink.exists():
        return {}
    rows = [json.loads(line) for line in sink.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {r["id"]: r for r in rows if r.get("prompt_version") == PROMPT_VERSION and r.get("verdict") is not None}


async def evaluate_cache(settings: Settings, *, verify: bool = True, sink: Path | None = None) -> SuiteResult:
    from samadhan.evaluation.runner import _persist, _retry, _tokens_total

    pairs = cache_pairs()
    emb = FastEmbedEmbeddings(EmbeddingModels(settings.retrieval))
    cached_vecs = await emb.aembed_documents([p.cached for p in pairs])
    sims = [_cos(c, await emb.aembed_query(p.query)) for p, c in zip(pairs, cached_vecs, strict=True)]

    summary: dict[str, float] = {}
    for t in THRESHOLDS:
        for name, value in _rates(pairs, [s >= t for s in sims]).items():
            summary[f"embedding@{t:.2f}.{name}"] = value
    # The lowest threshold with zero false hits: what "threshold only" would have to use.
    safe = next(
        (
            t
            for t in np.arange(0.70, 1.0, 0.005)
            if not any(s >= t for p, s in zip(pairs, sims, strict=True) if not p.same)
        ),
        1.0,
    )
    summary["embedding.min_safe_threshold"] = round(float(safe), 3)
    summary["embedding.hit_rate_at_min_safe"] = _rates(pairs, [s >= safe for s in sims])["paraphrase_hit_rate"]

    rows: list[dict[str, Any]] = []
    threshold = settings.retrieval.cache_candidate_threshold
    verifier = ModelRegistry(settings.llm).structured("fast", CacheVerdict) if verify else None
    # Resume: verdicts already persisted for this prompt version are reused (free-tier quotas end runs).
    done = _previous_verdicts(sink)
    hits: list[bool] = []
    latencies: list[float] = []
    tokens_before = _tokens_total()
    config: Any = {"callbacks": [TokenUsageCallback()]}
    for pair, sim in zip(pairs, sims, strict=True):
        candidate = sim >= threshold
        verdict: bool | None = None
        if candidate and normalize(pair.cached) == normalize(pair.query):
            verdict = True
        elif candidate and pair.id in done:
            verdict = done[pair.id]["verdict"]
        elif candidate and verifier is not None:
            started = time.perf_counter()
            prompt = CACHE_EQUIVALENCE.format(company=settings.agent.company_name, cached=pair.cached, new=pair.query)
            result: CacheVerdict = await _retry(lambda prompt=prompt: verifier.ainvoke(prompt, config))  # type: ignore[misc,arg-type]
            latencies.append(time.perf_counter() - started)
            verdict = result.same_answer
        hit = bool(candidate and verdict)
        hits.append(hit)
        row = {**pair.model_dump(), "similarity": round(sim, 4), "candidate": candidate, "verdict": verdict,
               "hit": hit, "prompt_version": PROMPT_VERSION}  # fmt: skip
        rows.append(row)
        if pair.id not in done:
            _persist(sink, row)

    if verify:
        for name, value in _rates(pairs, hits).items():
            summary[f"verified.{name}"] = value
        summary["verified.verifier_calls"] = float(len(latencies))
        summary["verified.verifier_p50_ms"] = round(float(np.median(latencies)) * 1000, 1) if latencies else 0.0
        summary["verified.tokens_per_verification"] = (
            round((_tokens_total() - tokens_before) / len(latencies), 1) if latencies else 0.0
        )
    summary["similarity.paraphrase_mean"] = mean([s for p, s in zip(pairs, sims, strict=True) if p.same])
    summary["similarity.near_miss_mean"] = mean([s for p, s in zip(pairs, sims, strict=True) if p.kind == "near_miss"])
    return SuiteResult(
        "cache" if verify else "cache-embedding",
        summary,
        rows,
        meta={"pairs": len(pairs), "candidate_threshold": threshold, "model": settings.retrieval.dense_model},
    )
