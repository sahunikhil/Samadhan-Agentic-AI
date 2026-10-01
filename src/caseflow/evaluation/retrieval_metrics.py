"""Classic information-retrieval metrics (no LLM, deterministic, free - ideal for CI).

We evaluate at the *document* level: a query is answered well if chunks from the
labeled relevant articles are ranked high. Chunk hits are collapsed to their
article in rank order (first occurrence wins).

* **hit@k**    - is at least one relevant article in the top k? ("did we find it at all")
* **recall@k** - fraction of the relevant articles found in the top k.
* **MRR**      - 1 / rank of the first relevant article. Rewards putting it *first*.
* **nDCG@k**   - discounted cumulative gain: every relevant article counts, with
  log-decaying credit by position, normalized by the ideal ranking.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def dedupe_docs(doc_ids: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(doc_ids))


def hit_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    return 1.0 if any(d in relevant for d in ranked[:k]) else 0.0


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 1.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def reciprocal_rank(ranked: Sequence[str], relevant: set[str]) -> float:
    for i, d in enumerate(ranked, start=1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 1) for i, d in enumerate(ranked[:k], start=1) if d in relevant)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def score_ranking(ranked_chunks_doc_ids: Sequence[str], relevant: Sequence[str], k: int = 5) -> dict[str, float]:
    ranked = dedupe_docs(ranked_chunks_doc_ids)
    rel = set(relevant)
    return {
        "hit@1": hit_at_k(ranked, rel, 1),
        f"hit@{k}": hit_at_k(ranked, rel, k),
        f"recall@{k}": recall_at_k(ranked, rel, k),
        "mrr": reciprocal_rank(ranked, rel),
        f"ndcg@{k}": ndcg_at_k(ranked, rel, k),
    }
