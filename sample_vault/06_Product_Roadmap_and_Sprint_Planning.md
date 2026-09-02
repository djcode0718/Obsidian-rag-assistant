---
title: Q3-Q4 Product Roadmap & Sprint Milestones
tags: [product, roadmap, sprint, milestones, agile]
updated: 2026-08-30
author: Lead Product Manager
---

# Q3-Q4 Product Roadmap & Sprint Milestones

## Strategic Vision
Transform isolated personal knowledge graphs (Obsidian vaults) into high-recall conversational intelligence engines that respect privacy and operate on zero-cost infrastructure.

## Sprint 24 Key Deliverables (Completed)
- **Local Dense Embeddings**: Integrated HuggingFace `sentence-transformers/all-MiniLM-L6-v2` generating 384d vectors on CPU.
- **Persistent Vector Storage**: Embedded ChromaDB engine operating entirely from local storage with cosine similarity indexing.
- **Provider Swapping**: Seamless runtime toggle between Groq LLaMA-3.3-70B and Google Gemini 1.5/2.5 Flash.
- **Grounded Attribution**: Interactive UI cards highlighting exact source notes, section headers, and context snippets.

## Future Milestones (Q4)
1. **Graph-Augmented RAG**: Parsing `[[wikilinks]]` in markdown notes to build topological knowledge graphs and perform multi-hop graph traversal retrieval.
2. **Hybrid BM25 + Dense Reranking**: Incorporating cross-encoder rerankers (`ms-marco-MiniLM-L-6-v2`) to boost precision on domain-specific acronyms.
3. **Local LLM Mode**: Optional Ollama integration for air-gapped, zero-network environments.
