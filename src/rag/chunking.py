from __future__ import annotations
from typing import List, Dict


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 100) -> List[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be < chunk_size")

    text = text or ""
    text = text.strip()
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = end - overlap

    return chunks


def chunk_docs(
    docs: List[Dict],
    chunk_size: int = 400,
    overlap: int = 100,
    start_chunk_id: int = 0
) -> List[Dict]:
    """
    Expected docs format:
      {"text": "...", "source": "file.pdf", "page": 1}
    Returns:
      [{"chunk_id": 1, "text": "...", "source": "...", "page": 1}, ...]
    """
    chunked: List[Dict] = []
    chunk_id = start_chunk_id

    for d in docs:
        text = (d.get("text") or "").strip()
        if not text:
            continue

        pieces = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        for piece in pieces:
            chunk_id += 1
            chunked.append({
                "chunk_id": chunk_id,
                "text": piece,
                "source": d.get("source"),
                "page": d.get("page"),
            })

    return chunked