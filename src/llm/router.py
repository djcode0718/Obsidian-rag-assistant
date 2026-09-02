"""Multi-model LLM Router with automatic fallback chain across Groq and Gemini."""

from __future__ import annotations

from typing import List, Optional, Tuple, Dict, Any

from src.llm.base import BaseLLMClient, LLMResponse, LLMError
from src.llm.groq_client import GroqClient
from src.llm.gemini_client import GeminiClient


class LLMRouter(BaseLLMClient):
    """Orchestrates an ordered fallback chain across Groq and Gemini models.

    Chain Order:
    1. Groq  -> llama-3.3-70b-versatile (Primary high-speed reasoning)
    2. Groq  -> openai/gpt-oss-120b (Intra-provider Groq fallback)
    3. Gemini -> gemini-2.5-flash (Inter-provider high throughput)
    4. Gemini -> gemini-1.5-flash (Final reliable fallback)
    """

    FALLBACK_CHAIN: List[Tuple[str, str]] = [
        ("groq", "llama-3.3-70b-versatile"),
        ("groq", "openai/gpt-oss-120b"),
        ("gemini", "gemini-2.5-flash"),
        ("gemini", "gemini-1.5-flash"),
    ]

    def __init__(
        self,
        groq_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        custom_chain: Optional[List[Tuple[str, str]]] = None,
    ) -> None:
        self.groq_api_key = (groq_api_key or "").strip()
        self.gemini_api_key = (gemini_api_key or "").strip()
        self.chain = custom_chain or list(self.FALLBACK_CHAIN)
        self._last_successful_model: Optional[str] = None
        self._last_successful_provider: Optional[str] = None

    @property
    def provider_name(self) -> str:
        return self._last_successful_provider or "LLM Router"

    @property
    def model_name(self) -> str:
        return self._last_successful_model or "auto-fallback-chain"

    def is_configured(self) -> bool:
        """Returns True if at least one provider in the chain has an API key."""
        return bool(self.groq_api_key or self.gemini_api_key)

    def update_keys(self, groq_key: Optional[str] = None, gemini_key: Optional[str] = None) -> None:
        """Updates provider credentials dynamically."""
        if groq_key is not None:
            self.groq_api_key = groq_key.strip()
        if gemini_key is not None:
            self.gemini_api_key = gemini_key.strip()

    def get_chain_status(self) -> List[Dict[str, Any]]:
        """Returns the readiness status of each model in the fallback chain."""
        status_list = []
        for provider, model in self.chain:
            is_ready = bool(self.groq_api_key) if provider == "groq" else bool(self.gemini_api_key)
            status_list.append({
                "provider": provider.capitalize(),
                "model": model,
                "is_configured": is_ready,
            })
        return status_list

    def _create_client(self, provider: str, model: str) -> BaseLLMClient:
        """Instantiates a client for a specific provider and model."""
        if provider == "groq":
            return GroqClient(api_key=self.groq_api_key, model=model)
        elif provider == "gemini":
            return GeminiClient(api_key=self.gemini_api_key, model=model)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
    ) -> LLMResponse:
        """Tries each (provider, model) pair in order until one succeeds.

        Each individual client handles a 1-time automatic backoff retry internally.
        If a model fails even after retry (or errors with 404/quota), this router
        captures the failure and advances to the next candidate model in the chain.

        Raises:
            LLMError: If all candidate models in the chain fail or are unconfigured.
        """
        if not self.is_configured():
            raise LLMError(
                "No LLM providers are configured. Please enter a Groq or Gemini API key in the sidebar.",
                provider="LLMRouter",
            )

        attempted_errors: List[str] = []

        for provider, model in self.chain:
            # Skip provider if its API key is not configured
            key_available = bool(self.groq_api_key) if provider == "groq" else bool(self.gemini_api_key)
            if not key_available:
                attempted_errors.append(f"{provider.capitalize()} ({model}): Skipped (missing API key)")
                continue

            try:
                client = self._create_client(provider, model)
                response = client.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                )

                self._last_successful_provider = response.provider
                self._last_successful_model = response.model
                return response

            except Exception as e:
                err_summary = str(e)
                # Keep error message concise for reporting
                if len(err_summary) > 120:
                    err_summary = err_summary[:117] + "..."
                attempted_errors.append(f"{provider.capitalize()} ({model}): {err_summary}")
                continue

        # All models failed
        formatted_failures = "\n".join(f"- {err}" for err in attempted_errors)
        raise LLMError(
            f"All models in the automatic fallback chain failed:\n{formatted_failures}\n\n"
            "Please check your API keys or rate limits in the sidebar.",
            provider="LLMRouter",
            is_rate_limit=any("429" in e or "rate limit" in e.lower() or "quota" in e.lower() for e in attempted_errors),
        )
