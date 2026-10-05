"""Knowledge-base loading and chunking.

Chunking strategy (and why)
---------------------------
1. **Structure-aware split** on Markdown headings (H1/H2/H3). Help-center articles
   are organised by topic, so a heading is the most natural semantic boundary -
   much better than cutting every N characters through the middle of a table.
2. **Size-bounded split** of long sections with overlap, so no chunk exceeds the
   embedding model's sweet spot (~256-512 tokens for bge-small).
3. **Contextual chunk headers**: the text we *embed* is prefixed with
   ``"<article title> > <section path>"``. A chunk that only says "The fee is
   waived for members" becomes findable for "laptop restocking fee" because its
   header says *Returns and Refunds Policy > Restocking fee*. This is a zero-cost
   variant of "contextual retrieval" (no LLM call per chunk).

Chunk IDs are deterministic (UUIDv5 of doc id + position) and every chunk carries
the SHA-256 of its source document, so re-ingestion is idempotent and only
re-embeds documents that actually changed.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

CHUNKER_VERSION = "2"  # 2: product scope in contextual headers
_NAMESPACE = uuid.UUID("5f3c2a8e-9f1b-4c7e-8a55-0b7d4f3e2c11")


@dataclass(slots=True)
class KBDocument:
    doc_id: str
    title: str
    category: str
    url: str
    body: str
    source_path: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        # The chunker version is part of the hash: changing how chunks are built re-indexes every
        # document on the next (incremental) ingest instead of silently keeping stale chunks.
        return hashlib.sha256(f"{CHUNKER_VERSION}\n{self.body}".encode()).hexdigest()


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    category: str
    url: str
    section: str
    text: str  # shown to the LLM and the user
    embed_text: str  # what we embed (text + contextual header)
    position: int
    content_hash: str
    product: str | None = None

    def payload(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "title": self.title,
            "category": self.category,
            "url": self.url,
            "section": self.section,
            "text": self.text,
            "position": self.position,
            "content_hash": self.content_hash,
            "product": self.product,
        }


def parse_markdown(path: Path) -> KBDocument:
    raw = path.read_text(encoding="utf-8")
    meta: dict[str, Any] = {}
    body = raw
    if raw.startswith("---"):
        _, front, body = raw.split("---", 2)
        meta = yaml.safe_load(front) or {}
    return KBDocument(
        doc_id=str(meta.get("id") or path.stem),
        title=str(meta.get("title") or path.stem.replace("-", " ").title()),
        category=str(meta.get("category") or "general"),
        url=str(meta.get("url") or ""),
        body=body.strip(),
        source_path=path.as_posix(),
        metadata=meta,
    )


def load_knowledge_base(kb_dir: Path) -> list[KBDocument]:
    docs = [parse_markdown(p) for p in sorted(kb_dir.glob("*.md"))]
    if not docs:
        raise FileNotFoundError(f"No markdown articles found in {kb_dir.resolve()}")
    return docs


ChunkStrategy = Literal["markdown", "fixed"]


def chunk_document(
    doc: KBDocument,
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    strategy: ChunkStrategy = "markdown",
    contextual_header: bool = True,
) -> list[Chunk]:
    """``strategy`` / ``contextual_header`` exist for the chunking ablation (``samadhan eval chunking``);
    production uses the defaults that the ablation selected."""
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")], strip_headers=True
    )
    size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", ". ", " ", ""]
    )
    if strategy == "markdown":
        sections = [
            (" > ".join(v for k, v in d.metadata.items() if k in ("h2", "h3") and v) or "Overview", d.page_content)
            for d in header_splitter.split_text(doc.body)
        ]
    else:  # fixed-size windows over the raw article, blind to its structure
        sections = [("", doc.body)]
    chunks: list[Chunk] = []
    position = 0
    for section_path, content in sections:
        for piece in size_splitter.split_text(content):
            text = piece.strip()
            if len(text) < 20:  # drop empty/trivial fragments
                continue
            scope = f" (applies to: {doc.metadata['product']})" if doc.metadata.get("product") else ""
            header = f"{doc.title}{scope} > {section_path}" if section_path else f"{doc.title}{scope}"
            chunks.append(
                Chunk(
                    chunk_id=str(uuid.uuid5(_NAMESPACE, f"{doc.doc_id}:{position}")),
                    doc_id=doc.doc_id,
                    title=doc.title,
                    category=doc.category,
                    url=doc.url,
                    section=section_path or "Overview",
                    text=text,
                    embed_text=f"{header}\n\n{text}" if contextual_header else text,
                    position=position,
                    content_hash=doc.content_hash,
                    product=doc.metadata.get("product"),
                )
            )
            position += 1
    return chunks


def chunk_documents(docs: list[KBDocument], *, chunk_size: int, chunk_overlap: int, **options: Any) -> list[Chunk]:
    return [c for d in docs for c in chunk_document(d, chunk_size=chunk_size, chunk_overlap=chunk_overlap, **options)]
