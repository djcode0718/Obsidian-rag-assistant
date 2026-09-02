# Obsidian Vault RAG Knowledge Assistant

> An enterprise-grade, privacy-respecting Retrieval-Augmented Generation (RAG) assistant designed for personal knowledge graphs (Obsidian markdown vaults). Operates entirely on free-tier, low-latency services with verifiable source citations and automatic cloud deployment resilience.

---

## Architecture Overview

```
                               ┌──────────────────────────────────────────────┐
                               │             Streamlit Web Interface          │
                               │  - Chat history & citation cards             │
                               │  - LLM provider toggle (Groq / Gemini)       │
                               │  - Real-time vault index metrics             │
                               └──────────────────────┬───────────────────────┘
                                                      │
                       ┌──────────────────────────────┴──────────────────────────────┐
                       │                                                             │
                       ▼                                                             ▼
           [1. Ingestion Pipeline]                                      [2. Query & Retrieval Pipeline]
  ┌─────────────────────────────────────────┐                  ┌─────────────────────────────────────────┐
  │  Source: sample_vault/ or uploaded .zip │                  │ User Question: Natural language query   │
  └────────────────────┬────────────────────┘                  └────────────────────┬────────────────────┘
                       │                                                            │
                       ▼                                                            ▼
  ┌─────────────────────────────────────────┐                  ┌─────────────────────────────────────────┐
  │ loader.py: Extract YAML frontmatter,    │                  │ embedder.py: Query vector generated via │
  │ titles, headings, and sanitization      │                  │ sentence-transformers (all-MiniLM-L6-v2)│
  └────────────────────┬────────────────────┘                  └────────────────────┬────────────────────┘
                       │                                                            │
                       ▼                                                            ▼
  ┌─────────────────────────────────────────┐                  ┌─────────────────────────────────────────┐
  │ chunker.py: Header-aware chunking       │                  │ chroma_store.py: Cosine distance search │
  │ (~500 tokens, 50-token overlap)         │                  │ Top-k nearest context chunks retrieved  │
  └────────────────────┬────────────────────┘                  └────────────────────┬────────────────────┘
                       │                                                            │
                       ▼                                                            ▼
  ┌─────────────────────────────────────────┐                  ┌─────────────────────────────────────────┐
  │ embedder.py: Local 384d dense vectors   │                  │ pipeline.py: Grounded prompt assembly   │
  │ (CPU/MPS, no paid API calls)            │                  │ with strict attribution constraints     │
  └────────────────────┬────────────────────┘                  └────────────────────┬────────────────────┘
                       │                                                            │
                       ▼                                                            ▼
  ┌─────────────────────────────────────────┐                  ┌─────────────────────────────────────────┐
  │ chroma_store.py: Persistent ChromaDB    │                  │ LLM Provider (Swappable with 1x retry): │
  │ collection saved to ./data/chroma_db    │                  │ - Groq LLaMA-3.3-70B (Primary, >300 t/s)│
  └─────────────────────────────────────────┘                  │ - Google Gemini Flash (Alternative)     │
                                                               └────────────────────┬────────────────────┘
                                                                                    │
                                                                                    ▼
                                                               ┌─────────────────────────────────────────┐
                                                               │ Grounded response with inline citations │
                                                               │ and expandable source cards             │
                                                               └─────────────────────────────────────────┘
```

---

## Approach & Technologies (Job Submission Summary)

*This section summarizes the design philosophy and technical stack for evaluation:*

- **Problem Formulation**: Personal knowledge bases in Obsidian contain highly contextual, interconnected notes with headers and YAML metadata. Naive fixed-character chunking tears headers from their context and induces hallucinations.
- **Header-Aware Chunking**: We engineered a parser that splits markdown by semantic headers (`#` through `####`) first. Any section exceeding ~500 tokens undergoes sliding-window token chunking with a 50-token overlap, while prepending the document title and section heading to each chunk to retain global document hierarchy.
- **Zero-Cost, Local Dense Embeddings**: Embeddings run 100% locally via `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, normalized cosine space). This eliminates embedding API costs, removes external network hops during indexing, and guarantees data privacy.
- **Local Persistent Vector Store**: Indexed into ChromaDB (`PersistentClient`) with cosine similarity. Metadata is preserved across all chunks (source note, heading, token count, file path).
- **Swappable Multi-Provider LLM Tiering**:
  - **Primary**: Groq API utilizing `llama-3.3-70b-versatile` running on Groq LPUs for instant, sub-second responses.
  - **Alternative / Fallback**: Google Gemini Flash via the official `google-generativeai` SDK.
  - **Fault Resilience**: 1-time automatic backoff retry on transient errors or HTTP 429 rate limits, followed by a graceful non-crashing UI warning banner if both attempts fail.
- **Ephemeral Cloud Readiness**: Automatic first-load detection auto-indexes the bundled `sample_vault/` if the vector database is uninitialized, ensuring reviewers on Streamlit Cloud or Hugging Face Spaces are greeted with a fully operational demo immediately.

---

## Technology Stack

| Layer | Component / Technology | Justification |
|---|---|---|
| **UI Framework** | [Streamlit](https://streamlit.io/) | Fast, reactive Python web UI with native chat elements |
| **Embeddings** | `sentence-transformers` (`all-MiniLM-L6-v2`) | Free, local 384d dense vectors; zero cloud API cost |
| **Vector Database** | [ChromaDB](https://www.trychroma.com/) | Lightweight, local, embedded persistent vector database |
| **Primary LLM** | Groq (`llama-3.3-70b-versatile`) | Ultra-fast inference (>300 t/s) on Groq LPU free tier |
| **Alternative LLM**| Google Gemini Flash (`google-generativeai`) | Reliable fallback with large context window |
| **Environment** | Python 3.11 + Conda (`obsidian-env1`) | Strict dependency isolation and deterministic builds |
| **Config / Secrets**| `python-dotenv` | Clean `.env` loading with OS environment variable precedence |

---

## Project Structure

```
.
├── app.py                         # Streamlit interactive application entrypoint
├── environment.yml                # Conda environment specification
├── requirements.txt               # Pip dependency manifest (for Cloud deployments)
├── .env                           # Local environment variables (git-ignored)
├── .env.example                   # Template environment file with blank keys
├── .gitignore                     # Git rules preventing secret and cache commits
├── README.md                      # Full technical documentation
├── src/
│   ├── __init__.py
│   ├── config.py                  # Environment loader, model configurations, and constants
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── loader.py              # Vault reader for directories & uploaded .zip archives
│   │   └── chunker.py             # Header-aware markdown chunker with metadata preservation
│   ├── embeddings/
│   │   ├── __init__.py
│   │   └── embedder.py            # Local sentence-transformers embedding wrapper
│   ├── vectorstore/
│   │   ├── __init__.py
│   │   └── chroma_store.py        # Persistent ChromaDB client and cosine search
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── base.py                # Abstract LLM client interface & error classes
│   │   ├── groq_client.py         # Groq client with 1-time rate limit backoff retry
│   │   └── gemini_client.py       # Google Gemini client (google-generativeai)
│   └── rag/
│       ├── __init__.py
│       └── pipeline.py            # Retrieval, prompt assembly, and citation synthesis
├── sample_vault/                  # Realistic Obsidian notes for instant evaluation
│   ├── 01_System_Architecture.md
│   ├── 02_Database_and_Storage_Strategy.md
│   ├── 03_LLM_Orchestration_and_Prompting.md
│   ├── 04_Security_and_Authentication.md
│   ├── 05_DevOps_and_CI_CD_Pipeline.md
│   └── 06_Product_Roadmap_and_Sprint_Planning.md
└── tests/
    └── test_pipeline.py           # Unit and integration test suite
```

---

## Local Setup Instructions

### 1. Prerequisites
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Miniforge](https://github.com/conda-forge/miniforge) installed.

### 2. Create and Activate the Conda Environment
```bash
# Clone the repository
git clone <your-repo-url>
cd "Obsidian Vault RAG Knowledge Assistant"

# Create conda environment named obsidian-env1 with Python 3.11
conda env create -f environment.yml

# Activate the environment
conda activate obsidian-env1
```

*(Alternatively, to install manually via conda and pip):*
```bash
conda create -y -n obsidian-env1 python=3.11
conda activate obsidian-env1
pip install -r requirements.txt
```

### 3. Configure API Keys
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Open `.env` and fill in your free API keys:
```bash
GROQ_API_KEY=gsk_your_groq_api_key_here
GEMINI_API_KEY=AIzaSy_your_gemini_api_key_here
```
> **Note**: You can obtain free API keys from:
> - Groq: [https://console.groq.com/keys](https://console.groq.com/keys)
> - Google AI Studio (Gemini): [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
>
> *If you prefer not to create a `.env` file, you can also paste your API key directly into the Streamlit sidebar at runtime.*

### 4. Run the Application Locally
```bash
streamlit run app.py
```
Open your browser to `http://localhost:8501`. On initial launch, the assistant automatically indexes `sample_vault/` and is ready for immediate questions.

### 5. Running the Test Suite
```bash
python -m pytest tests/test_pipeline.py -v
```

---

## Cloud Deployment Guide

### A. Streamlit Community Cloud
1. Push this repository to GitHub (ensure `.env` is **not** committed; it is ignored by `.gitignore`).
2. Log into [Streamlit Community Cloud](https://share.streamlit.io/) and create a **New app**.
3. Point to your repository, branch (`main`), and set Main file path to `app.py`.
4. In **Advanced Settings > Secrets**, paste your API keys:
   ```toml
   GROQ_API_KEY = "gsk_..."
   GEMINI_API_KEY = "AIzaSy..."
   ```
5. Click **Deploy**. On first boot, the app will auto-index the sample vault and run seamlessly.

### B. Hugging Face Spaces
1. Create a new Space on [Hugging Face Spaces](https://huggingface.co/spaces) selecting the **Streamlit** SDK.
2. Push the repository contents to the Space git remote.
3. In **Settings > Variables and secrets**, add `GROQ_API_KEY` and `GEMINI_API_KEY`.
4. The space will install dependencies from `requirements.txt` and start `app.py`.

---

## Key Design Decisions & Rationale

1. **Chunking Strategy**:
   Obsidian notes are heavily formatted with hierarchical markdown headers. Standard chunkers split indiscriminately across headings, losing context. Our `chunker.py` uses header-aware parsing to preserve heading trees, while keeping chunks around ~500 tokens with 50-token sliding overlap to fit comfortably within LLM context windows.

2. **Top-K Retrieval Parameter ($k = 4$)**:
   $k=4$ chunks provides ~1500 to 2000 tokens of high-relevance source context. This provides sufficient evidence to synthesize complex answers while maintaining low latency and avoiding context saturation.

3. **Grounding & Hallucination Prevention**:
   The prompt template enforces a zero-extrapolation policy. If the retrieved context does not contain the answer, the model explicitly responds: *"I cannot find sufficient evidence in your notes to answer this question."* Furthermore, all claims must be cited using the format `[Filename: Heading]`.

4. **1-Time Automatic Retry on Transient/Rate Limits**:
   Both Groq and Gemini free tiers occasionally encounter transient spikes or HTTP 429 rate limit errors. Instead of immediately failing, the client waits 2.0 seconds and retries once automatically. If the failure persists, a clean warning banner alerts the user and suggests toggling to the alternative provider.

5. **Auto-Ingestion on First Load**:
   Free cloud hosting services have ephemeral disks. If a container starts with an empty vector database, `app.py` automatically detects this and triggers background ingestion of `sample_vault/` before the user interacts with the app.

---

## License
MIT License. Free for educational, evaluation, and production use.
