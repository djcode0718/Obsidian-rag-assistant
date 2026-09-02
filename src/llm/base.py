"""Base abstraction for swappable LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


class LLMError(Exception):
    """Custom exception raised on unrecoverable LLM API errors."""

    def __init__(self, message: str, provider: str, is_rate_limit: bool = False) -> None:
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.is_rate_limit = is_rate_limit


@dataclass
class LLMResponse:
    """Standardized response container returned by LLM clients."""

    text: str
    provider: str
    model: str
    tokens_used: Optional[int] = None


class BaseLLMClient(ABC):
    """Abstract interface that all provider clients must implement."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the LLM provider (e.g. 'Groq', 'Google Gemini')."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Active model identifier."""
        pass

    @abstractmethod
    def is_configured(self) -> bool:
        """Returns True if the required credentials/API keys are available."""
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
    ) -> LLMResponse:
        """Generates a text completion for the supplied prompt.

        Args:
            prompt: Formatted user prompt with context.
            system_prompt: Optional system persona or constraints.
            temperature: Sampling temperature (default 0.2 for strict grounding).

        Returns:
            LLMResponse containing generated text.

        Raises:
            LLMError: If invocation fails or credentials are missing.
        """
        pass
