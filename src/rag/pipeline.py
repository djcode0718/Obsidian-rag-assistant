"""RAG Pipeline: Ingestion, Semantic Retrieval, Grounded Prompt Assembly, and Generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, BinaryIO

from src.config import config
from src.ingestion.loader import load_notes_from_directory, load_notes_from_zip, MarkdownNote
from src.ingestion.chunker import chunk_notes, MarkdownChunk
from src.embeddings.embedder import get_embedder, LocalEmbedder
from src.vectorstore.chroma_store import ChromaStore, RetrievedChunk
from src.llm.base import BaseLLMClient, LLMResponse, LLMError
from src.llm.groq_client import GroqClient
from src.llm.gemini_client import GeminiClient
from src.llm.router import LLMRouter


@dataclass
class Citation:
    """Attribution metadata for a source chunk grounding an answer."""

    source_file: str
    relative_path: str
    note_title: str
    heading: str
    similarity_score: float
    excerpt: str
    chunk_id: str


@dataclass
class RAGResult:
    """Structured response from the RAG pipeline."""

    question: str
    answer: str
    citations: List[Citation]
    provider: str
    model: str
    tokens_used: Optional[int] = None
    retrieved_chunks_count: int = 0


class RAGPipeline:
    """End-to-end RAG system for Obsidian Markdown Vaults."""

    def __init__(
        self,
        vector_store: Optional[ChromaStore] = None,
        embedder: Optional[LocalEmbedder] = None,
        groq_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
    ) -> None:
        self.vector_store = vector_store or ChromaStore(
            persist_directory=config.chroma_persist_dir,
            collection_name=config.collection_name,
        )
        self.embedder = embedder or get_embedder(model_name=config.embedding_model_name)

        # Initialize provider clients
        g_key = groq_api_key if groq_api_key is not None else config.groq_api_key
        gem_key = gemini_api_key if gemini_api_key is not None else config.gemini_api_key

        self.groq_client = GroqClient(api_key=g_key, model=config.groq_model)
        self.gemini_client = GeminiClient(api_key=gem_key, model=config.gemini_model)
        self.router = LLMRouter(groq_api_key=g_key, gemini_api_key=gem_key)

    def update_keys(self, groq_key: Optional[str] = None, gemini_key: Optional[str] = None) -> None:
        """Dynamically reconfigures API keys passed from the Streamlit UI."""
        if groq_key is not None:
            self.groq_client.api_key = groq_key.strip()
            self.groq_client._client = None
        if gemini_key is not None:
            self.gemini_client.api_key = gemini_key.strip()
            self.gemini_client._model_obj = None
        self.router.update_keys(groq_key=groq_key, gemini_key=gemini_key)

    def ingest_vault(
        self,
        source: str | Path | BinaryIO | bytes,
        is_zip: bool = False,
        reset_existing: bool = True,
    ) -> Dict[str, Any]:
        """Ingests markdown notes, chunks them, computes embeddings, and indexes in ChromaDB.

        Chunking Strategy:
            We segment notes on markdown headings (# through ####) to retain semantic cohesion.
            Any section exceeding ~500 tokens is split using a sliding window with 50 tokens
            overlap. Every chunk is prepended with its parent note title and active heading.

        Args:
            source: Path to vault directory, or zip bytes/file object.
            is_zip: True if source is a zip archive.
            reset_existing: If True, resets the Chroma collection before indexing.

        Returns:
            Dictionary of ingestion statistics.
        """
        if reset_existing:
            self.vector_store.reset_collection()

        # Step 1: Load notes
        if is_zip:
            notes: List[MarkdownNote] = load_notes_from_zip(source)
        else:
            notes: List[MarkdownNote] = load_notes_from_directory(source)

        if not notes:
            return {"num_notes": 0, "num_chunks": 0, "files": []}

        # Step 2: Header-aware chunking
        chunks: List[MarkdownChunk] = chunk_notes(
            notes,
            chunk_size_tokens=config.chunk_size_tokens,
            chunk_overlap_tokens=config.chunk_overlap_tokens,
        )

        if not chunks:
            return {"num_notes": len(notes), "num_chunks": 0, "files": [n.filename for n in notes]}

        # Step 3: Local Dense Embeddings (sentence-transformers)
        chunk_texts = [c.text for c in chunks]
        embeddings = self.embedder.embed_documents(chunk_texts)

        # Step 4: Persistent Indexing
        indexed_count = self.vector_store.add_chunks(chunks, embeddings)

        return {
            "num_notes": len(notes),
            "num_chunks": indexed_count,
            "files": [n.filename for n in notes],
        }

    def get_llm_client(self, provider: str = "auto") -> BaseLLMClient:
        """Resolves the requested LLM client by provider name or returns LLMRouter."""
        provider_clean = (provider or "auto").strip().lower()
        if provider_clean in ["auto", "router", "chain"]:
            return self.router
        elif "gemini" in provider_clean:
            return self.gemini_client
        elif "groq" in provider_clean:
            return self.groq_client
        return self.router

    def build_prompt(self, question: str, retrieved_chunks: List[RetrievedChunk]) -> tuple[str, str]:
        """Assembles a strict grounding prompt with numbered source contexts.

        Prompt Design Decision:
            - Explicit constraint: Must answer using ONLY provided context.
            - Explicit refusal instruction: If not present, declare lack of evidence.
            - Format requirement: Annotate claims with citation tags [Filename: Heading].
        """
        system_prompt = (
            "You are an expert AI Knowledge Assistant for a user's personal Obsidian Vault.\n"
            "Your job is to answer the user's questions based EXCLUSIVELY on the retrieved notes provided below.\n\n"
            "CRITICAL RULES:\n"
            "1. Ground every claim directly in the provided Context Chunks.\n"
            "2. If the context does not contain the answer, say:\n"
            "   'I cannot find sufficient evidence in your notes to answer this question.'\n"
            "   Do not make up facts or extrapolate beyond what is documented.\n"
            "3. Cite your sources inline using the format: [Filename: Heading] (e.g., [01_System_Architecture.md: Core Architectural Pillars]).\n"
            "4. Keep your answer clear, structured, and informative with bullet points where appropriate."
        )

        context_blocks = []
        for i, chunk in enumerate(retrieved_chunks, start=1):
            block = (
                f"--- CONTEXT CHUNK {i} ---\n"
                f"Source File: {chunk.source_file}\n"
                f"Note Title: {chunk.note_title}\n"
                f"Section Heading: {chunk.heading}\n"
                f"Similarity: {chunk.similarity_score:.2f}\n"
                f"Content:\n{chunk.text}\n"
            )
            context_blocks.append(block)

        context_str = "\n".join(context_blocks)
        user_prompt = (
            f"Here is the context retrieved from my Obsidian vault:\n\n"
            f"{context_str}\n\n"
            f"Question: {question}\n\n"
            f"Please provide a grounded answer based strictly on the above context, including citations:"
        )

        return system_prompt, user_prompt

    def query(
        self,
        question: str,
        provider: str = "auto",
        top_k: int = 4,
        temperature: float = 0.2,
    ) -> RAGResult:
        """Executes full RAG flow: Embed Query -> Retrieve -> Build Prompt -> LLM Call.

        Args:
            question: User natural language inquiry.
            provider: 'groq' or 'gemini'.
            top_k: Number of chunks to retrieve (default 4).
            temperature: Generation temperature (default 0.2 for factual fidelity).

        Returns:
            RAGResult containing answer, citations, and metadata.
        """
        # 1. Embed query
        query_vector = self.embedder.embed_query(question)

        # 2. Retrieve top-k nearest chunks
        retrieved = self.vector_store.query_by_embedding(query_vector, top_k=top_k)

        # 3. Handle empty collection or no chunks found
        if not retrieved:
            return RAGResult(
                question=question,
                answer=(
                    "Your vault has no notes indexed yet or no relevant content was found. "
                    "Please index the sample vault or upload notes from the sidebar."
                ),
                citations=[],
                provider=provider,
                model="None",
                retrieved_chunks_count=0,
            )

        # 4. Assemble prompt
        system_prompt, user_prompt = self.build_prompt(question, retrieved)

        # 5. Resolve LLM client
        llm = self.get_llm_client(provider)

        # 6. Generate answer
        response: LLMResponse = llm.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
        )

        # 7. Build citation objects with clean preview excerpts
        citations: List[Citation] = []
        for ch in retrieved:
            # Produce a clean excerpt without the heading banner
            lines = [l for l in ch.text.splitlines() if not l.startswith("## [")]
            excerpt = " ".join(lines).strip()
            if len(excerpt) > 280:
                excerpt = excerpt[:277] + "..."

            citations.append(
                Citation(
                    source_file=ch.source_file,
                    relative_path=ch.relative_path,
                    note_title=ch.note_title,
                    heading=ch.heading,
                    similarity_score=ch.similarity_score,
                    excerpt=excerpt,
                    chunk_id=ch.chunk_id,
                )
            )

        return RAGResult(
            question=question,
            answer=response.text,
            citations=citations,
            provider=response.provider,
            model=response.model,
            tokens_used=response.tokens_used,
            retrieved_chunks_count=len(retrieved),
        )
