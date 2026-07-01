"""
RAG router — /api/v1/rag

Endpoints:
  POST   /rag/documents          — upload and index a document
  GET    /rag/documents          — list all indexed documents
  GET    /rag/documents/{id}     — single document with chunk count
  DELETE /rag/documents/{id}     — remove from index and database
  POST   /rag/query              — query all documents
  POST   /rag/query/{doc_id}     — query a specific document
"""

from __future__ import annotations

import os
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from auth.dependencies import get_current_user
from fastapi import Depends
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from database import get_db
from models.rag_document import RAGDocument
from rag.pipeline import RAGPipeline, RAGConfig
from schemas.common import DataResponse, PaginatedResponse, make_pagination_meta

router = APIRouter(prefix="/rag", tags=["rag"], dependencies=[Depends(get_current_user)])


# ── Shared pipeline instance ──────────────────────────────────────────────────

_pipeline: Optional[RAGPipeline] = None


def get_pipeline() -> RAGPipeline:
    """Returns the module-level RAGPipeline, creating it once at first call."""
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline(
            vector_store_dir=os.path.join(settings.dataset_storage_path, "..", "rag"),
            config=RAGConfig(),
        )
    return _pipeline


# ── Response schemas ──────────────────────────────────────────────────────────

class RAGDocumentOut(BaseModel):
    id: int
    document_id: str
    name: str
    filename: str
    chunk_strategy: str
    chunk_count: int
    status: str
    error_message: Optional[str] = None
    dataset_id: Optional[int] = None
    created_at: str

    model_config = {"from_attributes": True}


class RAGQueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")


class CitationOut(BaseModel):
    document_id: str
    chunk_index: int
    source_name: str


class RAGQueryResponse(BaseModel):
    question: str
    answer: str
    citations: list[CitationOut]
    retrieved_chunks: list[dict]   # chunk text + score for transparency
    model: str
    input_tokens: int
    output_tokens: int
    error: Optional[str] = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _doc_out(doc: RAGDocument) -> RAGDocumentOut:
    return RAGDocumentOut(
        id=doc.id,
        document_id=doc.document_id,
        name=doc.name,
        filename=doc.filename,
        chunk_strategy=doc.chunk_strategy,
        chunk_count=doc.chunk_count,
        status=doc.status,
        error_message=doc.error_message,
        dataset_id=doc.dataset_id,
        created_at=doc.created_at.isoformat(),
    )


# ── Allowed extensions ────────────────────────────────────────────────────────

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".markdown", ".rst"}
MAX_DOC_BYTES = 50 * 1024 * 1024   # 50 MB per document


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/documents", response_model=DataResponse[RAGDocumentOut], status_code=201)
async def upload_rag_document(
    file: UploadFile = File(...),
    name: str = Form(default=""),
    chunk_strategy: str = Form(default="paragraph"),
    dataset_id: Optional[int] = Form(default=None),
    db: AsyncSession = Depends(get_db),
):
    """
    Uploads and indexes a document for RAG querying.

    Supported formats: PDF, DOCX, TXT, Markdown, RST.
    The document is chunked, embedded, and stored in ChromaDB.
    Returns immediately with status='indexing', then updates to 'ready'.
    """
    filename = file.filename or "document"
    ext = os.path.splitext(filename)[1].lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported format '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    if chunk_strategy not in ("paragraph", "fixed", "sentence"):
        raise HTTPException(
            status_code=422,
            detail="chunk_strategy must be 'paragraph', 'fixed', or 'sentence'",
        )

    content = await file.read(MAX_DOC_BYTES + 1)
    if len(content) > MAX_DOC_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Document exceeds {MAX_DOC_BYTES // (1024*1024)} MB limit.",
        )

    doc_name = name.strip() or os.path.splitext(filename)[0]
    document_id = f"rag-{uuid.uuid4().hex[:16]}"

    # Save file to disk
    import aiofiles
    rag_dir = os.path.join(settings.dataset_storage_path, "..", "rag_files")
    os.makedirs(rag_dir, exist_ok=True)
    file_path = os.path.join(rag_dir, f"{document_id}{ext}")

    async with aiofiles.open(file_path, "wb") as f:
        await f.write(content)

    # Create DB record with 'indexing' status
    doc = RAGDocument(
        document_id=document_id,
        name=doc_name,
        filename=filename,
        file_path=file_path,
        chunk_strategy=chunk_strategy,
        chunk_count=0,
        dataset_id=dataset_id,
        status="indexing",
    )
    db.add(doc)
    await db.flush()
    await db.refresh(doc)

    # Index in background via the job queue — gives retries on transient
    # embedding failures and bounds concurrency (unbounded asyncio.create_task
    # would let N simultaneous uploads all hit the embedding model at once).
    from jobs.queue import get_job_queue
    queue = get_job_queue()
    job_id = await queue.enqueue(
        _ingest_background, document_id, file_path, chunk_strategy, doc.id,
        max_attempts=2,   # one retry — embedding calls can transiently fail
    )

    return DataResponse(data=_doc_out(doc))


async def _ingest_background(
    document_id: str,
    file_path: str,
    chunk_strategy: str,
    db_id: int,
) -> dict:
    """Background task: indexes the document and updates the DB record."""
    from database import AsyncSessionLocal

    pipeline = get_pipeline()
    pipeline.config.chunk_strategy = chunk_strategy
    result = await pipeline.ingest(file_path=file_path, document_id=document_id)

    # Update DB record in a new session (the original request session may be closed)
    async with AsyncSessionLocal() as session:
        doc = await session.get(RAGDocument, db_id)
        if doc:
            if result.succeeded:
                doc.chunk_count = result.chunk_count
                doc.status = "ready"
            else:
                doc.status = "error"
                doc.error_message = result.error
            await session.commit()

    if not result.succeeded:
        # Raising lets the job queue retry — a transient embedding API
        # failure gets one more attempt instead of permanently failing.
        raise RuntimeError(result.error or "RAG ingestion failed")

    return {"chunk_count": result.chunk_count}


@router.get("/documents", response_model=PaginatedResponse[RAGDocumentOut])
async def list_rag_documents(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    """Lists all indexed RAG documents."""
    from sqlalchemy import func, select

    stmt = select(RAGDocument).order_by(RAGDocument.created_at.desc())
    total = await db.scalar(select(func.count()).select_from(stmt.subquery()))
    rows = await db.scalars(stmt.offset((page - 1) * page_size).limit(page_size))

    return PaginatedResponse(
        data=[_doc_out(d) for d in rows.all()],
        pagination=make_pagination_meta(page, page_size, total or 0),
    )


@router.get("/documents/{document_id}", response_model=DataResponse[RAGDocumentOut])
async def get_rag_document(document_id: str, db: AsyncSession = Depends(get_db)):
    """Returns metadata for a single indexed document."""
    result = await db.execute(
        select(RAGDocument).where(RAGDocument.document_id == document_id)
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found")
    return DataResponse(data=_doc_out(doc))


@router.delete("/documents/{document_id}", status_code=204)
async def delete_rag_document(document_id: str, db: AsyncSession = Depends(get_db)):
    """Removes a document from the vector index and the database."""
    result = await db.execute(
        select(RAGDocument).where(RAGDocument.document_id == document_id)
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found")

    # Delete from ChromaDB
    pipeline = get_pipeline()
    pipeline.delete_document(document_id)

    # Delete file from disk
    if doc.file_path and os.path.exists(doc.file_path):
        try:
            os.remove(doc.file_path)
        except OSError:
            pass

    await db.delete(doc)


@router.post("/query", response_model=DataResponse[RAGQueryResponse])
async def query_all_documents(
    body: RAGQueryRequest,
    db: AsyncSession = Depends(get_db),
):
    """Queries across all indexed RAG documents."""
    return await _run_query(body.question, body.k, document_ids=None)


@router.post("/query/{document_id}", response_model=DataResponse[RAGQueryResponse])
async def query_single_document(
    document_id: str,
    body: RAGQueryRequest,
    db: AsyncSession = Depends(get_db),
):
    """Queries a specific document only."""
    result = await db.execute(
        select(RAGDocument).where(RAGDocument.document_id == document_id)
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found")
    if doc.status != "ready":
        raise HTTPException(
            status_code=422,
            detail=f"Document is not ready (status: {doc.status})",
        )
    return await _run_query(body.question, body.k, document_ids=[document_id])


async def _run_query(
    question: str, k: int, document_ids: Optional[list[str]]
) -> DataResponse:
    """Shared query logic used by both query endpoints."""
    pipeline = get_pipeline()
    result = await pipeline.query(question=question, k=k, document_ids=document_ids)

    return DataResponse(data=RAGQueryResponse(
        question=question,
        answer=result.answer,
        citations=[
            CitationOut(
                document_id=c.document_id,
                chunk_index=c.chunk_index,
                source_name=c.source_name,
            )
            for c in result.citations
        ],
        retrieved_chunks=[
            {
                "text": c.text[:300] + ("…" if len(c.text) > 300 else ""),
                "score": round(c.final_score, 4),
                "document_id": c.document_id,
                "chunk_index": c.chunk_index,
                "source": c.metadata.get("source", c.document_id),
            }
            for c in result.chunks
        ],
        model=result.model,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        error=result.error,
    ))


# ── Evaluation endpoint ───────────────────────────────────────────────────────

class EvalCaseIn(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    ground_truth: Optional[str] = Field(None, max_length=5000)


class EvalRequest(BaseModel):
    cases: list[EvalCaseIn] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Test questions. Max 50 per run to control API costs.",
    )
    document_ids: Optional[list[str]] = Field(
        None,
        description="Restrict retrieval to these documents. None = all documents.",
    )
    k: int = Field(default=5, ge=1, le=20)


class EvalSampleOut(BaseModel):
    question: str
    answer: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    n_chunks_retrieved: int
    error: Optional[str] = None


class EvalReportOut(BaseModel):
    n_samples: int
    n_errors: int
    faithfulness_mean: float
    answer_relevancy_mean: float
    context_precision_mean: float
    overall_mean: float
    samples: list[EvalSampleOut]


@router.post("/evaluate", response_model=DataResponse[EvalReportOut])
async def evaluate_rag(body: EvalRequest):
    """
    Evaluates the RAG pipeline on a set of test questions.

    Computes three metrics per question:
      - faithfulness:      Are answers grounded in retrieved context? (LLM judge)
      - answer_relevancy:  Does the answer address the question? (embedding sim)
      - context_precision: Are retrieved chunks actually relevant? (LLM judge)

    Requires ANTHROPIC_API_KEY for faithfulness and context_precision.
    Answer relevancy works without an API key.

    Cost estimate: ~$0.005 per question with Claude Haiku.
    For 10 questions: ~$0.05. For 50 questions: ~$0.25.
    """
@router.post("/evaluate", response_model=DataResponse[EvalReportOut])
async def evaluate_rag(
    body: EvalRequest,
    current_user=Depends(get_current_user),
):
    from rag.evaluator import RAGEvaluator, EvalCase
    from auth.key_manager import get_anthropic_key

    api_key = await get_anthropic_key(current_user, require=False) or ""
    pipeline = get_pipeline()
    evaluator = RAGEvaluator(max_concurrency=3)

    cases = [EvalCase(question=c.question, ground_truth=c.ground_truth)
             for c in body.cases]

    report = await evaluator.evaluate(
        pipeline=pipeline,
        cases=cases,
        document_ids=body.document_ids,
        k=body.k,
        api_key=api_key,
    )

    return DataResponse(data=EvalReportOut(
        n_samples=report.n_samples,
        n_errors=report.n_errors,
        faithfulness_mean=report.faithfulness_mean,
        answer_relevancy_mean=report.answer_relevancy_mean,
        context_precision_mean=report.context_precision_mean,
        overall_mean=report.overall_mean,
        samples=[
            EvalSampleOut(
                question=s.question,
                answer=s.answer,
                faithfulness=s.faithfulness,
                answer_relevancy=s.answer_relevancy,
                context_precision=s.context_precision,
                n_chunks_retrieved=s.n_chunks_retrieved,
                error=s.error,
            )
            for s in report.samples
        ],
    ))

