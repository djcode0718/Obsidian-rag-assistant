---
title: Database and Vector Storage Strategy
tags: [database, storage, chromadb, postgresql, caching]
updated: 2026-08-20
author: Data Engineering Lead
---

# Database and Vector Storage Strategy

## Polyglot Storage Tiering
We reject one-size-fits-all database paradigms in favor of workload-tailored data engines:
- **Relational Operational Data**: PostgreSQL 16 with write-ahead logging (WAL) archiving to S3, housing transactional user profiles and billing data.
- **Vector Embeddings Store**: ChromaDB running locally in persistent mode (`PersistentClient`), storing 384-dimensional dense vectors generated from markdown note chunks.
- **In-Memory Cache**: Redis 7 cluster caching session states, rate-limit buckets, and hot semantic search query results with a 15-minute TTL.

## ChromaDB Indexing Configuration
For our markdown knowledge base:
1. **Space Metric**: Cosine distance (`hnsw:space: cosine`), normalized for unit vector dot product comparison.
2. **HNSW Hyperparameters**:
   - `M = 16` (number of bi-directional links per node)
   - `ef_construction = 100` (search depth during index construction)
   - `ef_search = 50` (search depth during query time)
3. **Persistence Directory**: Local filesystem volume mounted at `./data/chroma_db`, ensuring survival across container restarts without remote network hops.

## Backup and Disaster Recovery
- PostgreSQL snapshots run every 4 hours with 30-day point-in-time recovery (PITR).
- ChromaDB collections are backed up via volume snapshots before major document re-indexing jobs.
