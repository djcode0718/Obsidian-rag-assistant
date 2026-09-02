"""Embeddings package for local sentence-transformers."""

from src.embeddings.embedder import LocalEmbedder, get_embedder

__all__ = ["LocalEmbedder", "get_embedder"]
