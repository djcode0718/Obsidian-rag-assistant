---
title: System Architecture Overview
tags: [architecture, backend, microservices, event-driven]
updated: 2026-08-15
author: Staff Systems Architect
---

# System Architecture Overview

## Executive Summary
Our enterprise platform is built upon a loosely coupled, event-driven microservices topology. The primary objective is delivering sub-50ms p99 latency for analytical workflows while maintaining fault tolerance across multi-region deployments.

## Core Architectural Pillars
1. **Decoupled Asynchrony**: Asynchronous message passing for all non-blocking state mutations via Apache Kafka.
2. **Polyglot Persistence**: Matching data models to specialized storage engines (see [[02_Database_and_Storage_Strategy]]).
3. **Graceful Degradation**: Multi-tier LLM fallbacks and circuit breakers to absorb external provider outages (see [[03_LLM_Orchestration_and_Prompting]]).
4. **Zero-Trust Boundary**: Strict cryptographic identity verification across all internal and edge RPCs (see [[04_Security_and_Authentication]]).

## Ingestion & Knowledge Serving
The knowledge subsystem utilizes a dual-path pipeline:
- **Streaming Pipeline**: Real-time markdown document changes captured via webhooks and processed into embedding vectors.
- **Batch Reconciliation**: Nightly cron reconciliation ensuring vector index consistency across cold storage and live ChromaDB collections.
- **Retrieval Engine**: Hybrid lexical-dense vector search powered by local sentence-transformer models (`all-MiniLM-L6-v2`), avoiding cloud provider lock-in and cold-start latency.

## Service Mesh & Gateway
Edge ingress traffic terminates at an Envoy-powered API gateway that manages SSL termination, distributed rate limiting, and mTLS propagation down to internal Kubernetes pods.
