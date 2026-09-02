"""Configuration module for the Obsidian Vault RAG Knowledge Assistant.

Loads environment variables and exposes unified settings for models,
chunking strategies, and vector storage.
"""

from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

# Base directory of the repository
BASE_DIR = Path(__file__).resolve().parent.parent

# Load local .env if present (does not override existing system env vars)
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=False)


@dataclass
class AppConfig:
    """Application configuration container."""

    # Provider API Keys (read from environment or secrets)
    groq_api_key: str = os.getenv("GROQ_API_KEY", "").strip()
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "").strip()

    # LLM Models
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

    # Local Embeddings
    embedding_model_name: str = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

    # Vector Storage
    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", str(BASE_DIR / "data" / "chroma_db"))
    collection_name: str = os.getenv("CHROMA_COLLECTION_NAME", "obsidian_vault_notes")

    # Chunking & Retrieval Parameters
    chunk_size_tokens: int = int(os.getenv("CHUNK_SIZE_TOKENS", "500"))
    chunk_overlap_tokens: int = int(os.getenv("CHUNK_OVERLAP_TOKENS", "50"))
    default_top_k: int = int(os.getenv("DEFAULT_TOP_K", "4"))

    # Sample Vault Path
    sample_vault_path: Path = BASE_DIR / "sample_vault"

    def update_keys(self, groq_key: str | None = None, gemini_key: str | None = None) -> None:
        """Dynamically update keys if provided via Streamlit sidebar."""
        if groq_key is not None:
            self.groq_api_key = groq_key.strip()
        if gemini_key is not None:
            self.gemini_api_key = gemini_key.strip()


# Global singleton instance
config = AppConfig()
