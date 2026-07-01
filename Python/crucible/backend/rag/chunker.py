"""
RAG Chunker — text extraction and document splitting.

WHY CHUNKING MATTERS
--------------------
Embedding models have a context limit (typically 256–512 tokens).
Retrieval quality degrades when chunks are too large — you retrieve an
entire chapter when you need one paragraph. And when chunks are too small,
you retrieve isolated sentences without enough context to answer a question.

The sweet spot is roughly 300–600 words (1,500–3,000 characters) with
10–15% overlap between adjacent chunks.

THREE CHUNKING STRATEGIES
--------------------------
paragraph  — split at natural paragraph breaks (double newlines). Only
             falls back to mid-paragraph splitting when a paragraph exceeds
             max_chars. Best for structured documents: reports, docs, articles.

fixed      — split every max_chars characters with overlap_chars of overlap.
             Consistent, predictable chunk sizes regardless of document
             structure. Best for unstructured text.

sentence   — accumulate sentences until max_chars is reached, then start a
             new chunk with overlap_chars of trailing context. Best for
             dense technical text where every sentence carries independent
             information.

WHY OVERLAP
-----------
Without overlap, a sentence that falls exactly on a chunk boundary gets
split. Neither chunk alone is useful. Overlap ensures the boundary sentence
appears in full in at least one chunk.

TEXT EXTRACTION
---------------
PDF (pdfplumber): understands column layout, handles headers/footers,
preserves paragraph structure better than PyPDF2.

DOCX (python-docx): accesses individual paragraphs and preserves structure.

TXT / Markdown / RST: direct read, no processing needed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ── Output type ───────────────────────────────────────────────────────────────

@dataclass
class TextChunk:
    """One piece of a document ready for embedding and storage."""
    text: str
    chunk_index: int
    char_start: int
    char_end: int
    metadata: dict = field(default_factory=dict)

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", " ")
        return f"TextChunk(idx={self.chunk_index}, words={self.word_count}, preview={preview!r})"


# ── Text extractors ───────────────────────────────────────────────────────────

def extract_text(file_path: str) -> str:
    """
    Extracts plain text from a file based on its extension.
    Returns a single string with all pages / sections concatenated,
    separated by double newlines so the chunker sees them as paragraphs.
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext == ".pdf":
        return _extract_pdf(path)
    elif ext == ".docx":
        return _extract_docx(path)
    elif ext in (".txt", ".md", ".markdown", ".rst"):
        return path.read_text(encoding="utf-8", errors="replace")
    else:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Supported: .pdf, .docx, .txt, .md, .markdown, .rst"
        )


def _extract_pdf(path: Path) -> str:
    try:
        import pdfplumber
    except ImportError:
        raise ImportError("pdfplumber is required for PDF extraction: pip install pdfplumber")

    pages = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text and text.strip():
                pages.append(text.strip())
    return "\n\n".join(pages)


def _extract_docx(path: Path) -> str:
    try:
        from docx import Document
    except ImportError:
        raise ImportError("python-docx is required for DOCX extraction: pip install python-docx")

    doc = Document(str(path))
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


# ── Main entry point ──────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    strategy: str = "paragraph",
    max_chars: int = 1500,
    overlap_chars: int = 150,
    metadata: Optional[dict] = None,
) -> list[TextChunk]:
    """
    Splits text into chunks suitable for embedding and vector storage.

    Args:
        text:          The full document text (plain text, already extracted).
        strategy:      'paragraph' | 'fixed' | 'sentence'
        max_chars:     Maximum characters per chunk. Default 1500 ≈ 250 words.
                       all-MiniLM-L6-v2 / fastembed BAAI/bge-small-en-v1.5
                       both support up to ~512 tokens ≈ 2000 characters safely.
        overlap_chars: Characters of overlap between adjacent chunks.
                       Default 150 ≈ one sentence. Prevents clean-cut losses.
        metadata:      Base dict added to every chunk (e.g. {'source': 'report.pdf'}).

    Returns:
        List of TextChunk objects in document order.
    """
    meta = metadata or {}
    text = _clean_text(text)

    if not text:
        return []

    if strategy == "paragraph":
        raw = _chunk_by_paragraph(text, max_chars, overlap_chars)
    elif strategy == "fixed":
        raw = _chunk_fixed(text, max_chars, overlap_chars)
    elif strategy == "sentence":
        raw = _chunk_by_sentence(text, max_chars, overlap_chars)
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy!r}. "
                         f"Choose from: 'paragraph', 'fixed', 'sentence'")

    return [
        TextChunk(
            text=chunk_text_val,
            chunk_index=i,
            char_start=char_start,
            char_end=char_end,
            metadata={**meta, "chunk_index": i, "strategy": strategy},
        )
        for i, (chunk_text_val, char_start, char_end) in enumerate(raw)
        if chunk_text_val.strip()
    ]


# ── Text cleaning ─────────────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """
    Normalises whitespace without destroying paragraph structure.
    - Multiple blank lines → double newline (paragraph separator)
    - Multiple spaces / tabs → single space
    - Strip leading / trailing whitespace
    """
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


# ── Paragraph strategy ────────────────────────────────────────────────────────

def _chunk_by_paragraph(
    text: str, max_chars: int, overlap_chars: int
) -> list[tuple[str, int, int]]:
    """
    Splits at double-newline paragraph boundaries.

    Accumulates consecutive paragraphs until max_chars is reached.
    When a single paragraph exceeds max_chars, falls back to sentence
    splitting within that paragraph to stay within the limit.

    Returns list of (chunk_text, char_start, char_end).
    """
    paragraphs = re.split(r"\n\n+", text)
    chunks: list[tuple[str, int, int]] = []
    current = ""
    current_start = 0
    pos = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            pos += 2
            continue

        # Find this paragraph's actual position in the original text
        para_start = text.find(para, pos)
        if para_start == -1:
            para_start = pos
        pos = para_start + len(para)

        # Oversized single paragraph — recurse with sentence splitting
        if len(para) > max_chars:
            if current.strip():
                chunks.append((current.strip(), current_start, para_start))
                current = ""
            sub = _chunk_by_sentence(para, max_chars, overlap_chars)
            for s_text, s_start, s_end in sub:
                chunks.append((s_text, para_start + s_start, para_start + s_end))
            current_start = pos
            continue

        candidate = (current + "\n\n" + para).strip() if current else para

        if len(candidate) <= max_chars:
            if not current:
                current_start = para_start
            current = candidate
        else:
            if current.strip():
                chunks.append((current.strip(), current_start, para_start))
                overlap = current[-overlap_chars:] if len(current) > overlap_chars else current
                current = (overlap + "\n\n" + para).strip()
                current_start = max(0, para_start - len(overlap))
            else:
                current = para
                current_start = para_start

    if current.strip():
        chunks.append((current.strip(), current_start, len(text)))

    return chunks


# ── Fixed strategy ────────────────────────────────────────────────────────────

def _chunk_fixed(
    text: str, max_chars: int, overlap_chars: int
) -> list[tuple[str, int, int]]:
    """
    Splits every max_chars characters with overlap_chars of overlap.
    Simple, predictable, does not respect word or sentence boundaries.
    """
    chunks = []
    step = max(1, max_chars - overlap_chars)
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append((text[start:end], start, end))
        if end == len(text):
            break
        start += step
    return chunks


# ── Sentence strategy ─────────────────────────────────────────────────────────

def _chunk_by_sentence(
    text: str, max_chars: int, overlap_chars: int
) -> list[tuple[str, int, int]]:
    """
    Accumulates sentences until max_chars is reached.

    Sentence detection splits on '.', '!', '?' followed by whitespace and
    an uppercase letter. Handles common abbreviations (Mr., Dr., e.g.)
    adequately without requiring NLTK or spaCy.
    """
    # Split on sentence-ending punctuation + whitespace + uppercase
    raw_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z\"\'])', text)

    chunks: list[tuple[str, int, int]] = []
    current = ""
    current_start = 0
    pos = 0

    for sentence in raw_sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        s_start = text.find(sentence, pos)
        if s_start == -1:
            s_start = pos
        pos = s_start + len(sentence)

        candidate = (current + " " + sentence).strip() if current else sentence

        if len(candidate) <= max_chars:
            if not current:
                current_start = s_start
            current = candidate
        else:
            if current.strip():
                chunks.append((current.strip(), current_start, s_start))
                overlap = current[-overlap_chars:] if len(current) > overlap_chars else current
                current = (overlap + " " + sentence).strip()
                current_start = max(0, s_start - len(overlap))
            else:
                # Single sentence exceeds max_chars — include it whole
                chunks.append((sentence, s_start, pos))

    if current.strip():
        chunks.append((current.strip(), current_start, len(text)))

    return chunks


# ── Convenience: chunk a file end-to-end ─────────────────────────────────────

def chunk_file(
    file_path: str,
    strategy: str = "paragraph",
    max_chars: int = 1500,
    overlap_chars: int = 150,
) -> list[TextChunk]:
    """
    Extracts text from a file and chunks it in one call.
    Adds {'source': filename} to each chunk's metadata automatically.
    """
    path = Path(file_path)
    text = extract_text(file_path)
    return chunk_text(
        text,
        strategy=strategy,
        max_chars=max_chars,
        overlap_chars=overlap_chars,
        metadata={"source": path.name, "file_path": str(path)},
    )
