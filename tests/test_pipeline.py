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


def test_router_unconfigured_and_skipping():
    from src.llm.router import LLMRouter

    # Both blank
    router_empty = LLMRouter(groq_api_key="", gemini_api_key="")
    assert not router_empty.is_configured()
    with pytest.raises(LLMError) as exc_info:
        router_empty.generate("test prompt")
    assert "No LLM providers are configured" in str(exc_info.value)

    # Only Gemini configured
    router_gemini = LLMRouter(groq_api_key="", gemini_api_key="dummy_gemini_key")
    status = router_gemini.get_chain_status()
    assert len(status) == 4
    # Groq models skipped
    assert not status[0]["is_configured"]
    assert not status[1]["is_configured"]
    # Gemini models ready
    assert status[2]["is_configured"]
    assert status[3]["is_configured"]


def test_router_fallthrough_on_simulated_failure(monkeypatch):
    from src.llm.router import LLMRouter
    from src.llm.base import BaseLLMClient, LLMResponse, LLMError

    router = LLMRouter(groq_api_key="dummy_groq", gemini_api_key="dummy_gemini")

    call_history = []

    class MockFailingClient(BaseLLMClient):
        def __init__(self, provider, model):
            self._provider = provider
            self._model = model
        @property
        def provider_name(self): return self._provider
        @property
        def model_name(self): return self._model
        def is_configured(self): return True
        def generate(self, prompt, system_prompt=None, temperature=0.2):
            call_history.append((self._provider, self._model))
            raise LLMError("Simulated 429 Rate Limit", provider=self._provider, is_rate_limit=True)

    class MockSuccessClient(BaseLLMClient):
        def __init__(self, provider, model):
            self._provider = provider
            self._model = model
        @property
        def provider_name(self): return self._provider
        @property
        def model_name(self): return self._model
        def is_configured(self): return True
        def generate(self, prompt, system_prompt=None, temperature=0.2):
            call_history.append((self._provider, self._model))
            return LLMResponse(text="Success from fallback", provider=self._provider, model=self._model)

    def mock_create_client(self, provider, model):
        # Model 1 fails, Model 2 succeeds
        if model == "llama-3.3-70b-versatile":
            return MockFailingClient(provider, model)
        return MockSuccessClient(provider, model)

    monkeypatch.setattr(LLMRouter, "_create_client", mock_create_client)

    response = router.generate("What is our database strategy?")
    assert response.text == "Success from fallback"
    assert response.model == "openai/gpt-oss-120b"
    assert len(call_history) == 2
    assert call_history[0] == ("groq", "llama-3.3-70b-versatile")
    assert call_history[1] == ("groq", "openai/gpt-oss-120b")


def test_router_all_models_fail(monkeypatch):
    from src.llm.router import LLMRouter
    from src.llm.base import BaseLLMClient, LLMError

    router = LLMRouter(groq_api_key="dummy_groq", gemini_api_key="dummy_gemini")

    class MockAlwaysFailClient(BaseLLMClient):
        def __init__(self, provider, model):
            self._provider = provider
            self._model = model
        @property
        def provider_name(self): return self._provider
        @property
        def model_name(self): return self._model
        def is_configured(self): return True
        def generate(self, prompt, system_prompt=None, temperature=0.2):
            raise LLMError(f"{self._model} unavailable", provider=self._provider)

    monkeypatch.setattr(LLMRouter, "_create_client", lambda self, p, m: MockAlwaysFailClient(p, m))

    with pytest.raises(LLMError) as exc_info:
        router.generate("test question")
    assert "All models in the automatic fallback chain failed" in str(exc_info.value)

