"""Local persistent ChromaDB vector store wrapper."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.config import Settings


@dataclass
class RetrievedChunk:
    """Represents a retrieved chunk with similarity score and metadata."""

    chunk_id: str
    text: str
    source_file: str
    relative_path: str
    note_title: str
    heading: str
    similarity_score: float
    metadata: Dict[str, Any]


class ChromaStore:
    """Manages local, persistent ChromaDB collections."""

    def __init__(
        self,
        persist_directory: str | Path = "./data/chroma_db",
        collection_name: str = "obsidian_vault_notes",
    ) -> None:
        self.persist_directory = Path(persist_directory).resolve()
        self.collection_name = collection_name
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # Initialize persistent client
        self._client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False, is_persistent=True),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def collection(self):
        """Returns the active ChromaDB collection."""
        return self._collection

    def count(self) -> int:
        """Returns total number of chunks stored in the collection."""
        return self._collection.count()

    def is_empty(self) -> bool:
        """Checks if the collection has zero documents."""
        return self.count() == 0

    def reset_collection(self) -> None:
        """Clears and re-creates the collection."""
        try:
            self._client.delete_collection(name=self.collection_name)
        except Exception:
            pass
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(
        self,
        chunks: List[Any],
        embeddings: List[List[float]],
        batch_size: int = 250,
    ) -> int:
        """Stores chunks with their corresponding embeddings in batches.

        Args:
            chunks: List of MarkdownChunk instances.
            embeddings: Parallel list of float vectors.
            batch_size: Maximum items per Chroma insert call.

        Returns:
            Number of chunks successfully indexed.
        """
        if not chunks:
            return 0

        total = len(chunks)
        for i in range(0, total, batch_size):
            batch_chunks = chunks[i : i + batch_size]
            batch_embeds = embeddings[i : i + batch_size]

            ids = [c.chunk_id for c in batch_chunks]
            docs = [c.text for c in batch_chunks]
            metas = []
            for c in batch_chunks:
                # Chroma requires scalar values (str, int, float, bool)
                m = {
                    "source_file": str(c.source_file),
                    "relative_path": str(c.relative_path),
                    "note_title": str(c.note_title),
                    "heading": str(c.heading),
                    "chunk_index": int(c.chunk_index),
                    "token_count": int(c.token_count),
                }
                if "tags" in c.metadata:
                    m["tags"] = str(c.metadata["tags"])
                metas.append(m)

            self._collection.upsert(
                ids=ids,
                documents=docs,
                embeddings=batch_embeds,
                metadatas=metas,
            )

        return total

    def query_by_embedding(
        self,
        query_embedding: List[float],
        top_k: int = 4,
    ) -> List[RetrievedChunk]:
        """Queries the vector store for the nearest chunks using cosine similarity.

        Args:
            query_embedding: Dense embedding vector of user query.
            top_k: Number of nearest neighbors to retrieve.

        Returns:
            List of RetrievedChunk sorted by similarity (highest score first).
        """
        total_in_db = self.count()
        if total_in_db == 0:
            return []

        n_results = min(top_k, total_in_db)
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        retrieved: List[RetrievedChunk] = []
        if not results or not results["ids"] or not results["ids"][0]:
            return retrieved

        ids = results["ids"][0]
        docs = results["documents"][0] if results.get("documents") else [""] * len(ids)
        metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(ids)
        dists = results["distances"][0] if results.get("distances") else [0.0] * len(ids)

        for cid, doc, meta, dist in zip(ids, docs, metas, dists):
            # Cosine distance in Chroma is 1 - cosine_similarity (range 0 to 2)
            similarity = max(0.0, min(1.0, 1.0 - float(dist)))
            retrieved.append(
                RetrievedChunk(
                    chunk_id=cid,
                    text=doc,
                    source_file=meta.get("source_file", "Unknown"),
                    relative_path=meta.get("relative_path", "Unknown"),
                    note_title=meta.get("note_title", "Untitled"),
                    heading=meta.get("heading", ""),
                    similarity_score=round(similarity, 4),
                    metadata=meta,
                )
            )

        return retrieved

    def get_stats(self) -> Dict[str, Any]:
        """Returns high-level statistics about the stored collection."""
        total_chunks = self.count()
        if total_chunks == 0:
            return {
                "total_chunks": 0,
                "total_notes": 0,
                "note_files": [],
            }

        # Fetch metadatas to count unique notes
        data = self._collection.get(include=["metadatas"])
        metas = data.get("metadatas", [])
        unique_files = sorted({m.get("source_file") for m in metas if m and "source_file" in m})

        return {
            "total_chunks": total_chunks,
            "total_notes": len(unique_files),
            "note_files": unique_files,
        }
