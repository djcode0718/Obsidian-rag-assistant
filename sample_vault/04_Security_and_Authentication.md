---
title: Security, Authentication & Key Management
tags: [security, auth, jwt, rbac, secrets]
updated: 2026-08-25
author: Head of Information Security
---

# Security, Authentication & Key Management

## Identity Architecture
Our platform enforces strict identity boundaries:
- **JWT Authentication**: Short-lived JSON Web Tokens (15-minute expiration) signed via RS256 asymmetric keys.
- **Refresh Token Rotation**: Stored in HTTP-only secure cookies with family tracking to detect token reuse and replay attacks.
- **Role-Based Access Control (RBAC)**: Fine-grained permission attributes (`admin`, `editor`, `viewer`, `auditor`).

## API Key & Secrets Sanitation
1. **Never Commit Secrets**: All configuration files (`.env`) are strictly excluded from version control via `.gitignore`.
2. **Pre-commit Hooks**: Automated git hooks run regex scanning (`detect-secrets`, `gitleaks`) on every commit.
3. **Runtime Injection**: Production instances receive secrets strictly via container environment variables or cloud secrets managers (Streamlit Secrets, Hugging Face Secrets, AWS Secrets Manager).

## Vault Isolation & Sandboxing
Uploaded user zip archives containing markdown notes are uncompressed into isolated ephemeral sandboxes:
- Strict path sanitization preventing Directory Traversal attacks (e.g. rejecting `../` relative path escapes).
- File type whitelisting restricting ingestion strictly to `.md` and `.markdown` text files.
