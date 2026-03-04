from functools import lru_cache
from sentence_transformers import SentenceTransformer

@lru_cache(maxsize=2)
def get_embedder(model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2") -> SentenceTransformer:
    """
    Returns a cached SentenceTransformer so you don't reload the model multiple times.
    """
    return SentenceTransformer(model_name)