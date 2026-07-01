"""ORM model for RAG documents indexed in the vector store."""

from __future__ import annotations
from typing import Optional

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column

from models.base import Base, TimestampMixin


class RAGDocument(Base, TimestampMixin):
    """
    Tracks every document ingested into the RAG vector store.

    The actual chunk texts and embeddings live in ChromaDB — this table
    stores only the metadata needed to surface documents in the UI and
    to perform document-level operations (delete, list, chunk count).

    document_id is the ChromaDB collection identifier. It is generated
    by the router as a UUID and stored here so the UI can reference it.
    """

    __tablename__ = "rag_documents"

    id: Mapped[int] = mapped_column(sa.Integer, primary_key=True, autoincrement=True)

    # Unique identifier used as the ChromaDB collection key
    document_id: Mapped[str] = mapped_column(
        sa.String(64), unique=True, nullable=False, index=True
    )

    # Human-readable name (usually the original filename)
    name: Mapped[str] = mapped_column(sa.String(255), nullable=False)

    # Original filename for display and attribution in citations
    filename: Mapped[str] = mapped_column(sa.String(512), nullable=False)

    # Path to the stored file (same storage as datasets)
    file_path: Mapped[Optional[str]] = mapped_column(sa.String(1024), nullable=True)

    # Chunking settings used — stored so UI can show how it was processed
    chunk_strategy: Mapped[str] = mapped_column(sa.String(20), nullable=False, default="paragraph")
    chunk_count: Mapped[int] = mapped_column(sa.Integer, nullable=False, default=0)

    # Optional link to a dataset for domain-specific Q&A
    dataset_id: Mapped[Optional[int]] = mapped_column(
        sa.Integer,
        sa.ForeignKey("datasets.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # Lifecycle status
    status: Mapped[str] = mapped_column(
        sa.String(20), nullable=False, default="indexing"
    )
    # status values: indexing | ready | error

    error_message: Mapped[Optional[str]] = mapped_column(sa.Text, nullable=True)
