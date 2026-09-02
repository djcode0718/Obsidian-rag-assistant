"""Streamlit Entrypoint: Obsidian Vault RAG Knowledge Assistant.

A modular, production-ready Knowledge Assistant grounded in Obsidian Markdown vaults.
Uses local embeddings, ChromaDB, and swappable Groq / Gemini LLM providers.
"""

from __future__ import annotations

import os
from pathlib import Path
import streamlit as st

from src.config import config
from src.rag.pipeline import RAGPipeline, RAGResult, Citation
from src.llm.base import LLMError


# --- Page Configuration ---
st.set_page_config(
    page_title="Obsidian Vault RAG Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom Styling ---
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        background: linear-gradient(90deg, #6366f1, #a855f7, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
    }
    .citation-card {
        background-color: rgba(99, 102, 241, 0.05);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 8px;
        padding: 10px 14px;
        margin-top: 8px;
        margin-bottom: 8px;
    }
    .citation-badge {
        display: inline-block;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 4px;
        margin-right: 6px;
    }
    .badge-file {
        background-color: #312e81;
        color: #c7d2fe;
    }
    .badge-heading {
        background-color: #3b0764;
        color: #f5d0fe;
    }
    .badge-score {
        background-color: #064e3b;
        color: #a7f3d0;
    }
    .stAlert {
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --- Pipeline Caching & Initialization ---
@st.cache_resource(show_spinner=False)
def get_rag_pipeline() -> RAGPipeline:
    """Instantiates and caches the singleton RAG pipeline."""
    return RAGPipeline()


pipeline = get_rag_pipeline()

# Guarantee router is available on older cached instances without requiring server restart
if not hasattr(pipeline, "router") or pipeline.router is None:
    from src.llm.router import LLMRouter
    pipeline.router = LLMRouter(
        groq_api_key=getattr(pipeline.groq_client, "api_key", config.groq_api_key),
        gemini_api_key=getattr(pipeline.gemini_client, "api_key", config.gemini_api_key),
    )


# --- Auto-Ingest on First Startup (Amendment 2) ---
# Ephemeral hosting platforms (Streamlit Cloud, Hugging Face Spaces) may boot with
# an empty vector store. If the Chroma collection is empty, auto-ingest sample_vault/
# immediately so evaluators are never greeted by an unindexed, broken application.
if "auto_ingest_executed" not in st.session_state:
    st.session_state.auto_ingest_executed = True
    if pipeline.vector_store.is_empty():
        with st.status("⚡ First-time startup: Indexing sample Obsidian vault...", expanded=True) as status:
            st.write("📁 Reading markdown notes from `sample_vault/`...")
            st.write("🧠 Computing local dense embeddings with `sentence-transformers` (`all-MiniLM-L6-v2`)...")
            try:
                stats = pipeline.ingest_vault(
                    source=config.sample_vault_path,
                    is_zip=False,
                    reset_existing=True,
                )
                status.update(
                    label=f"✅ Indexed {stats['num_notes']} notes ({stats['num_chunks']} chunks) successfully!",
                    state="complete",
                    expanded=False,
                )
            except Exception as e:
                status.update(label=f"❌ Auto-indexing failed: {e}", state="error")


# --- Chat State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []


# --- Sidebar: Configuration & Controls ---
with st.sidebar:
    st.markdown("### 📚 Vault Knowledge Base")

    # Ingestion Source Selector
    vault_mode = st.radio(
        "Ingestion Source:",
        options=["Sample Vault (Built-in)", "Upload Vault (.zip)"],
        index=0,
        help="Use the bundled sample vault or upload your own exported Obsidian vault as a .zip file.",
    )

    uploaded_zip = None
    if vault_mode == "Upload Vault (.zip)":
        uploaded_zip = st.file_uploader(
            "Upload Obsidian Vault .zip",
            type=["zip"],
            help="Zip archive containing markdown (.md) notes.",
        )

    # Ingestion Trigger Button
    reindex_label = "Re-Index Sample Vault" if vault_mode == "Sample Vault (Built-in)" else "Index Uploaded Zip"
    if st.button(reindex_label, use_container_width=True, type="primary"):
        with st.spinner("Processing markdown files and generating local embeddings..."):
            try:
                if vault_mode == "Sample Vault (Built-in)":
                    stats = pipeline.ingest_vault(config.sample_vault_path, is_zip=False, reset_existing=True)
                else:
                    if uploaded_zip is None:
                        st.error("Please upload a .zip archive first.")
                        stats = None
                    else:
                        stats = pipeline.ingest_vault(uploaded_zip, is_zip=True, reset_existing=True)

                if stats:
                    st.success(f"Indexed {stats['num_notes']} notes into {stats['num_chunks']} chunks!")
                    st.rerun()
            except Exception as e:
                st.error(f"Ingestion failed: {e}")

    st.markdown("---")

    # Vector Store Statistics
    st.markdown("### 📊 Index Statistics")
    stats = pipeline.vector_store.get_stats()
    col_stat1, col_stat2 = st.columns(2)
    with col_stat1:
        st.metric("Notes Indexed", stats.get("total_notes", 0))
    with col_stat2:
        st.metric("Chunks Stored", stats.get("total_chunks", 0))

    with st.expander("Indexed Files", expanded=False):
        note_files = stats.get("note_files", [])
        if note_files:
            for nf in note_files:
                st.markdown(f"- `{nf}`")
        else:
            st.info("No files indexed yet.")

    st.markdown("---")

    # LLM Provider Configuration
    st.markdown("### 🤖 LLM Provider Settings")

    llm_mode = st.radio(
        "Execution Mode:",
        options=["Auto Fallback Chain (Recommended)", "Manual Model Selection"],
        index=0,
        help="Auto Fallback Chain tries all 4 models in order: Groq llama-3.3 -> Groq gpt-oss-120b -> Gemini 2.5 -> Gemini 1.5.",
    )

    if llm_mode == "Manual Model Selection":
        manual_choice = st.selectbox(
            "Select Single Model:",
            options=[
                "Groq (llama-3.3-70b-versatile)",
                "Groq (openai/gpt-oss-120b)",
                "Google Gemini (gemini-2.5-flash)",
                "Google Gemini (gemini-1.5-flash)",
            ],
            index=0,
        )
        if "Groq" in manual_choice:
            active_provider_key = "groq"
            if "gpt-oss-120b" in manual_choice:
                pipeline.groq_client.model = "openai/gpt-oss-120b"
            else:
                pipeline.groq_client.model = "llama-3.3-70b-versatile"
        else:
            active_provider_key = "gemini"
            if "gemini-1.5-flash" in manual_choice:
                pipeline.gemini_client.model = "gemini-1.5-flash"
            else:
                pipeline.gemini_client.model = "gemini-2.5-flash"
    else:
        active_provider_key = "auto"

    # API Keys Configuration (Never display raw secrets in UI values)
    env_groq_key = os.getenv("GROQ_API_KEY", "").strip()
    env_gemini_key = os.getenv("GEMINI_API_KEY", "").strip()

    st.markdown("##### API Key Configuration")

    groq_placeholder = "Configured via .env (enter to override)" if env_groq_key else "gsk_... (paste here)"
    groq_input = st.text_input(
        "Groq API Key:",
        value="",
        type="password",
        placeholder=groq_placeholder,
        help="Leave blank to use key from .env / environment variables.",
    )
    if env_groq_key and not groq_input:
        st.caption("🔒 *Loaded securely from environment (.env)*")

    gemini_placeholder = "Configured via .env (enter to override)" if env_gemini_key else "AIzaSy... (paste here)"
    gemini_input = st.text_input(
        "Gemini API Key:",
        value="",
        type="password",
        placeholder=gemini_placeholder,
        help="Leave blank to use key from .env / environment variables.",
    )
    if env_gemini_key and not gemini_input:
        st.caption("🔒 *Loaded securely from environment (.env)*")

    # Resolve effective keys: manual input takes precedence over environment
    effective_groq_key = groq_input.strip() if groq_input.strip() else env_groq_key
    effective_gemini_key = gemini_input.strip() if gemini_input.strip() else env_gemini_key

    # Sync resolved keys to pipeline and router
    pipeline.update_keys(groq_key=effective_groq_key, gemini_key=effective_gemini_key)

    # 4-Model Fallback Chain Key Status Panel
    st.markdown("##### ⛓️ 4-Model Fallback Status")
    chain_status = pipeline.router.get_chain_status()
    for idx, item in enumerate(chain_status, 1):
        status_label = "🟢 Ready" if item["is_configured"] else "🔴 Missing Key"
        st.caption(f"**[{idx}] {item['provider']}** `{item['model']}`: {status_label}")

    # Retrieval Tuning
    with st.expander("⚙️ Retrieval Parameters", expanded=False):
        top_k = st.slider("Top-K Chunks Retrieved:", min_value=1, max_value=8, value=4, step=1)
        temperature = st.slider("LLM Temperature:", min_value=0.0, max_value=1.0, value=0.2, step=0.05)


# --- Main Area: Header & Quick Prompts ---
st.markdown('<div class="main-header">Obsidian Vault RAG Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Grounded conversational question-answering with local embeddings and verified source citations.</div>',
    unsafe_allow_html=True,
)

# Render Quick Starter Questions if chat is empty
if not st.session_state.messages:
    st.markdown("##### 💡 Suggested Questions from the Sample Vault")
    sample_queries = [
        "What is our database and vector storage strategy?",
        "What are the grounding and hallucination prevention rules?",
        "What are our core architectural pillars?",
        "What key deliverables were completed in Sprint 24?",
    ]

    cols = st.columns(2)
    selected_sample = None
    for i, sq in enumerate(sample_queries):
        col = cols[i % 2]
        if col.button(f"🔍 {sq}", key=f"sq_{i}", use_container_width=True):
            selected_sample = sq


# --- Render Conversation History ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Render citations if available for assistant responses
        if msg.get("citations"):
            with st.expander(f"📑 Grounded Sources ({len(msg['citations'])} chunks)", expanded=False):
                for idx, cite in enumerate(msg["citations"], start=1):
                    st.markdown(
                        f"""
                        <div class="citation-card">
                            <span class="citation-badge badge-file">📄 {cite.source_file}</span>
                            <span class="citation-badge badge-heading">🔖 {cite.heading}</span>
                            <span class="citation-badge badge-score">🎯 Match: {cite.similarity_score * 100:.1f}%</span>
                            <p style="margin-top: 8px; margin-bottom: 2px; font-size: 0.88rem; color: #cbd5e1; font-style: italic;">
                                "{cite.excerpt}"
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        if msg.get("meta"):
            st.caption(f"🤖 Answered by **{msg['meta'].get('provider')}** ({msg['meta'].get('model')})")


# --- Handle User Input ---
user_input = st.chat_input("Ask a question about your Obsidian vault notes...")

# If a sample query button was clicked, trigger it as user_input
if not user_input and "selected_sample" in locals() and selected_sample:
    user_input = selected_sample

if user_input:
    # 1. Display and record user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. Check credentials for the chosen provider
    active_client = pipeline.get_llm_client(active_provider_key)

    with st.chat_message("assistant"):
        if not active_client.is_configured():
            warning_text = (
                f"⚠️ **{active_client.provider_name} API Key is not configured.**\n\n"
                f"Please enter your key in the left sidebar under **LLM Provider Settings**, "
                f"or populate the `.env` file with `{active_client.provider_name.upper()}_API_KEY=`.\n\n"
                f"*Both Groq and Google Gemini offer free API tiers.*"
            )
            st.warning(warning_text)
            st.session_state.messages.append({
                "role": "assistant",
                "content": warning_text,
                "citations": [],
            })
        else:
            with st.spinner(f"Retrieving notes & generating grounded answer via {active_client.provider_name}..."):
                try:
                    rag_result: RAGResult = pipeline.query(
                        question=user_input,
                        provider=active_provider_key,
                        top_k=top_k,
                        temperature=temperature,
                    )

                    # Display Answer
                    st.markdown(rag_result.answer)

                    # Display Citations
                    if rag_result.citations:
                        with st.expander(f"📑 Grounded Sources ({len(rag_result.citations)} chunks)", expanded=True):
                            for idx, cite in enumerate(rag_result.citations, start=1):
                                st.markdown(
                                    f"""
                                    <div class="citation-card">
                                        <span class="citation-badge badge-file">📄 {cite.source_file}</span>
                                        <span class="citation-badge badge-heading">🔖 {cite.heading}</span>
                                        <span class="citation-badge badge-score">🎯 Match: {cite.similarity_score * 100:.1f}%</span>
                                        <p style="margin-top: 8px; margin-bottom: 2px; font-size: 0.88rem; color: #cbd5e1; font-style: italic;">
                                            "{cite.excerpt}"
                                        </p>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                    st.caption(f"🤖 Answered by **{rag_result.provider}** ({rag_result.model})")

                    # Record in session history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": rag_result.answer,
                        "citations": rag_result.citations,
                        "meta": {"provider": rag_result.provider, "model": rag_result.model},
                    })

                except LLMError as err:
                    err_msg = (
                        f"⚠️ **{err.provider} API Notice**\n\n"
                        f"{err.message}\n\n"
                        f"*Tip: You can switch to the other provider using the sidebar dropdown.*"
                    )
                    st.warning(err_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": err_msg,
                        "citations": [],
                    })
                except Exception as exc:
                    st.error(f"Unexpected error during query processing: {exc}")
