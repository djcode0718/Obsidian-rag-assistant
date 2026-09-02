"""Local sentence-transformers embedding provider."""

from __future__ import annotations

from typing import List, Optional
import numpy as np


class LocalEmbedder:
    """Wraps sentence-transformers model (all-MiniLM-L6-v2) for local CPU/MPS inference."""

    _instance: Optional[LocalEmbedder] = None

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None

    def _ensure_loaded(self) -> None:
        """Lazy loader for the transformer model to avoid slow startup until needed."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load local embedding model '{self.model_name}': {e}\n"
                    "Ensure 'sentence-transformers' is installed in your conda environment."
                ) from e

    def embed_documents(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Computes dense embeddings for a list of document strings.

        Args:
            texts: List of text chunks to embed.
            batch_size: Batch size for inference.

        Returns:
            List of float vectors (each dimension 384 for all-MiniLM-L6-v2).
        """
        if not texts:
            return []

        self._ensure_loaded()
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """Computes a dense embedding for a single user query string.

        Args:
            query: Natural language query string.

        Returns:
            Single list of floats representing normalized query vector.
        """
        self._ensure_loaded()
        embedding = self._model.encode(
            query,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        if isinstance(embedding, np.ndarray):
            return embedding.tolist()
        return list(embedding)


# Cached singleton instance helper
_GLOBAL_EMBEDDER: Optional[LocalEmbedder] = None


def get_embedder(model_name: str = "all-MiniLM-L6-v2") -> LocalEmbedder:
    """Returns the shared singleton instance of LocalEmbedder."""
    global _GLOBAL_EMBEDDER
    if _GLOBAL_EMBEDDER is None or _GLOBAL_EMBEDDER.model_name != model_name:
        _GLOBAL_EMBEDDER = LocalEmbedder(model_name=model_name)
    return _GLOBAL_EMBEDDER
