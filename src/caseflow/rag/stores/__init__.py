"""Vector store backends behind one interface.

* ``QdrantHybridStore`` - purpose-built vector DB, native sparse vectors + server-side
  RRF fusion. Runs embedded (no server) for local dev, or as a service/cluster.
* ``PgVectorStore`` - pgvector + Postgres full-text search, fused with RRF in SQL.
  Lets a small deployment run everything (checkpoints, memory, vectors) in *one*
  Postgres - the cheapest production topology.
"""

from __future__ import annotations

from caseflow.config import RetrievalSettings, Settings
from caseflow.rag.stores.base import SearchHit, VectorStore


def create_vector_store(settings: Settings) -> VectorStore:
    r: RetrievalSettings = settings.retrieval
    if r.backend == "pgvector":
        from caseflow.rag.stores.pgvector import PgVectorStore

        if settings.persistence.database_url is None:
            raise ValueError("pgvector backend requires CASEFLOW_PERSISTENCE__DATABASE_URL")
        return PgVectorStore(settings.persistence.database_url.get_secret_value(), table=r.collection)
    from caseflow.rag.stores.qdrant import QdrantHybridStore

    return QdrantHybridStore(
        collection=r.collection,
        url=r.qdrant_url,
        path=r.qdrant_path,
        api_key=r.qdrant_api_key.get_secret_value() if r.qdrant_api_key else None,
    )


__all__ = ["SearchHit", "VectorStore", "create_vector_store"]
