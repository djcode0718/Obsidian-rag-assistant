---
title: LLM Orchestration and Grounding Guidelines
tags: [llm, prompt-engineering, groq, gemini, rag]
updated: 2026-08-22
author: Principal AI Researcher
---

# LLM Orchestration and Grounding Guidelines

## Dual Provider Tiering Model
To ensure maximum availability, cost efficiency, and ultra-low generation latency:
1. **Primary Provider (Groq)**:
   - Model: `llama-3.3-70b-versatile` running on Groq's custom LPU (Language Processing Unit) architecture.
   - Strengths: >300 tokens/second throughput, deterministic inference, high reasoning capability on grounded synthesis.
2. **Fallback / Alternative Provider (Google Gemini)**:
   - Model: `gemini-1.5-flash` (via `google-generativeai` SDK).
   - Strengths: Massive context window, exceptional retrieval resilience, reliable fallback when Groq hits tier limits.

## Grounding & Hallucination Prevention Rules
Every RAG prompt assembled by our orchestrator must adhere to strict contractual guidelines:
- **Zero Extrapolation**: If the retrieved context chunks do not contain sufficient factual evidence to answer the prompt, the model must explicitly respond: *"I cannot find sufficient evidence in your notes to answer this question."*
- **Verifiable Citations**: Every factual assertion must be attributed to a specific source document and section heading using the format `[Filename: Heading]`.
- **Top-K Retrieval Parameter**: Default `top_k = 4` chunks, providing ~1500 to 2000 tokens of high-relevance contextual anchor.

## Rate Limiting and Automatic Retry
Transient errors (HTTP 429, 503) are mitigated by:
- Immediate 1-time automatic retry after an exponential backoff jitter (2.0s).
- Graceful UI notification banners if both primary and secondary attempts fail, never hard-crashing the client interface.
