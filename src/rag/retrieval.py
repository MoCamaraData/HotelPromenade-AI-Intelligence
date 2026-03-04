import json
from pathlib import Path
from typing import List, Dict

import faiss
import numpy as np

from src.rag.embeddings import get_embedder


class FaissRetriever:
    def __init__(
        self,
        index_dir: str | Path,
        chunks_path: str | Path,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ):
        self.index_dir = Path(index_dir)
        self.chunks_path = Path(chunks_path)

        # Load FAISS index
        self.index = faiss.read_index(str(self.index_dir / "index.faiss"))

        # Load chunks (jsonl)
        self.chunks: List[Dict] = []
        with open(self.chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                self.chunks.append(json.loads(line))

        # Load embedding model (cached)
        self.embedder = get_embedder(model_name)

        print(f" Retriever ready | Vectors: {self.index.ntotal}")

    def retrieve(self, query: str, k: int = 3) -> List[Dict]:
        q = self.embedder.encode([query], normalize_embeddings=True)
        q = np.array(q, dtype="float32")

        scores, ids = self.index.search(q, k)

        results = []
        for rank, idx in enumerate(ids[0]):
            chunk = self.chunks[int(idx)]
            results.append({
                "score": float(scores[0][rank]),
                "text": chunk["text"],
                "source": chunk["source"],
                "page": chunk["page"],
            })

        return results