"""Sanity and integration tests for Obsidian Vault RAG Knowledge Assistant."""

import tempfile
from pathlib import Path
import pytest

from src.ingestion.loader import load_notes_from_directory, MarkdownNote
from src.ingestion.chunker import chunk_notes, count_tokens, MarkdownChunk
from src.embeddings.embedder import LocalEmbedder
from src.vectorstore.chroma_store import ChromaStore
from src.llm.base import LLMError
from src.llm.groq_client import GroqClient
from src.llm.gemini_client import GeminiClient
from src.rag.pipeline import RAGPipeline


@pytest.fixture
def sample_vault_dir():
    base = Path(__file__).resolve().parent.parent / "sample_vault"
    return base


def test_note_loader(sample_vault_dir):
    notes = load_notes_from_directory(sample_vault_dir)
    assert len(notes) >= 5, f"Expected at least 5 sample notes, got {len(notes)}"
    
    filenames = [n.filename for n in notes]
    assert "01_System_Architecture.md" in filenames
    assert "02_Database_and_Storage_Strategy.md" in filenames

    first = next(n for n in notes if n.filename == "01_System_Architecture.md")
    assert first.title == "System Architecture Overview"
    assert "tags" in first.metadata
    assert "architecture" in first.metadata["tags"]


def test_chunker(sample_vault_dir):
    notes = load_notes_from_directory(sample_vault_dir)
    chunks = chunk_notes(notes, chunk_size_tokens=300, chunk_overlap_tokens=30)
    
    assert len(chunks) >= len(notes)
    for chunk in chunks:
        assert isinstance(chunk, MarkdownChunk)
        assert chunk.source_file.endswith(".md")
        assert chunk.heading != ""
        assert chunk.text.startswith(f"## [{chunk.note_title}]")
        assert chunk.token_count > 0


def test_local_embedder():
    embedder = LocalEmbedder(model_name="all-MiniLM-L6-v2")
    q_embed = embedder.embed_query("How does vector indexing work?")
    assert len(q_embed) == 384, f"Expected 384 dimensions, got {len(q_embed)}"

    doc_embeds = embedder.embed_documents(["Sample document one", "Sample document two"])
    assert len(doc_embeds) == 2
    assert len(doc_embeds[0]) == 384


def test_chroma_store_and_retrieval(sample_vault_dir):
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ChromaStore(persist_directory=tmpdir, collection_name="test_vault")
        assert store.is_empty()
        assert store.count() == 0

        # Load and chunk
        notes = load_notes_from_directory(sample_vault_dir)
        chunks = chunk_notes(notes[:2], chunk_size_tokens=300)
        
        embedder = LocalEmbedder(model_name="all-MiniLM-L6-v2")
        embeddings = embedder.embed_documents([c.text for c in chunks])

        added = store.add_chunks(chunks, embeddings)
        assert added == len(chunks)
        assert store.count() == len(chunks)
        assert not store.is_empty()

        # Query
        query_vec = embedder.embed_query("PostgreSQL WAL archiving and ChromaDB cosine distance")
        results = store.query_by_embedding(query_vec, top_k=2)
        assert len(results) == 2
        assert results[0].source_file in ["01_System_Architecture.md", "02_Database_and_Storage_Strategy.md"]
        assert results[0].similarity_score > 0.0


def test_llm_clients_unconfigured():
    groq = GroqClient(api_key="")
    assert not groq.is_configured()
    with pytest.raises(LLMError) as exc_groq:
        groq.generate("test prompt")
    assert "Groq API Key is not configured" in str(exc_groq.value)

    gemini = GeminiClient(api_key="")
    assert not gemini.is_configured()
    with pytest.raises(LLMError) as exc_gem:
        gemini.generate("test prompt")
    assert "Gemini API Key is not configured" in str(exc_gem.value)


def test_rag_prompt_construction(sample_vault_dir):
    notes = load_notes_from_directory(sample_vault_dir)
    chunks = chunk_notes(notes[:2], chunk_size_tokens=300)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ChromaStore(persist_directory=tmpdir, collection_name="test_prompt")
        embedder = LocalEmbedder(model_name="all-MiniLM-L6-v2")
        embeddings = embedder.embed_documents([c.text for c in chunks])
        store.add_chunks(chunks, embeddings)

        pipeline = RAGPipeline(vector_store=store, embedder=embedder)
        retrieved = store.query_by_embedding(embedder.embed_query("Event-driven architecture"), top_k=2)
        sys_p, user_p = pipeline.build_prompt("Explain event streaming", retrieved)

        assert "CRITICAL RULES" in sys_p
        assert "CONTEXT CHUNK 1" in user_p
        assert "Explain event streaming" in user_p
