"""Groq API client implementation for high-speed inference."""

from __future__ import annotations

import time
from typing import Optional

from src.llm.base import BaseLLMClient, LLMResponse, LLMError


class GroqClient(BaseLLMClient):
    """Client for Groq cloud API utilizing high-throughput LPUs."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "llama-3.3-70b-versatile",
    ) -> None:
        self.api_key = (api_key or "").strip()
        self.model = model
        self._client = None

    @property
    def provider_name(self) -> str:
        return "Groq"

    @property
    def model_name(self) -> str:
        return self.model

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def _get_client(self):
        if self._client is None:
            if not self.is_configured():
                raise LLMError(
                    "Groq API Key is not configured. Please supply a key in the sidebar or via the GROQ_API_KEY environment variable.",
                    provider=self.provider_name,
                )
            try:
                from groq import Groq
                self._client = Groq(api_key=self.api_key)
            except Exception as e:
                raise LLMError(f"Failed to initialize Groq client: {e}", provider=self.provider_name) from e
        return self._client

    def _resolve_model(self, client) -> str:
        """Verifies or resolves an active model if the default is unavailable."""
        preferred_candidates = [
            self.model,
            "openai/gpt-oss-120b",
            "llama-3.3-70b-versatile",
            "openai/gpt-oss-20b",
            "qwen/qwen3.6-27b",
            "llama-3.1-8b-instant",
        ]
        try:
            available = {m.id for m in client.models.list().data}
            for candidate in preferred_candidates:
                if candidate in available:
                    self.model = candidate
                    return candidate
        except Exception:
            pass
        return self.model

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
    ) -> LLMResponse:
        """Executes completion with a 1-time retry on transient/rate-limit errors."""
        client = self._get_client()
        active_model = self._resolve_model(client)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        max_attempts = 2
        last_exception = None

        for attempt in range(1, max_attempts + 1):
            try:
                response = client.chat.completions.create(
                    model=active_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=1024,
                )

                choice = response.choices[0]
                raw_text = choice.message.content or ""

                # Cleanly strip <think>...</think> reasoning blocks if present
                import re
                clean_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
                if not clean_text:
                    clean_text = raw_text.strip()

                tokens = response.usage.total_tokens if response.usage else None

                return LLMResponse(
                    text=clean_text,
                    provider=self.provider_name,
                    model=active_model,
                    tokens_used=tokens,
                )

            except Exception as exc:
                last_exception = exc
                err_msg = str(exc).lower()
                is_rate_limit = "rate limit" in err_msg or "429" in err_msg or "quota" in err_msg
                is_transient = is_rate_limit or "connection" in err_msg or "timeout" in err_msg or "503" in err_msg

                if attempt < max_attempts and is_transient:
                    # Short backoff before retry to absorb temporary spikes
                    time.sleep(2.0)
                    continue
                else:
                    # Final attempt failed
                    clean_err = f"Groq Error: {str(exc)}"
                    if is_rate_limit:
                        clean_err = (
                            "Groq Rate Limit Exceeded (HTTP 429). The system attempted an automatic retry, "
                            "but the rate limit remains active. Please wait a few seconds or switch to Gemini Flash."
                        )
                    raise LLMError(clean_err, provider=self.provider_name, is_rate_limit=is_rate_limit) from exc

        raise LLMError(f"Groq invocation failed: {last_exception}", provider=self.provider_name)
