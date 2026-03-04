import json
from pathlib import Path
from typing import Tuple, List, Dict

import faiss
import numpy as np

from src.rag.embeddings import get_embedder


def build_and_save_faiss_index(
    chunks_path: str | Path,
    index_dir: str | Path,
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    batch_size: int = 64,
) -> Tuple[int, int]:
    """
    Build a FAISS index from chunks (.jsonl) and save:
      - index.faiss
      - chunks_meta.json
    Returns: (num_vectors, embedding_dim)
    """
    chunks_path = Path(chunks_path)
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    chunks: List[Dict] = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    texts = [c["text"] for c in chunks]

    embedder = get_embedder(model_name)
    emb = embedder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    emb = np.array(emb, dtype="float32")

    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)

    faiss.write_index(index, str(index_dir / "index.faiss"))
    with open(index_dir / "chunks_meta.json", "w", encoding="utf-8") as f:
        json.dump(
            [{"chunk_id": c["chunk_id"], "source": c["source"], "page": c["page"]} for c in chunks],
            f,
            ensure_ascii=False,
            indent=2,
        )

    return index.ntotal, dim