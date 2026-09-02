"""Google Gemini API client using the official google-generativeai SDK."""

from __future__ import annotations

import time
from typing import Optional

from src.llm.base import BaseLLMClient, LLMResponse, LLMError


class GeminiClient(BaseLLMClient):
    """Client for Google Gemini Flash using google-generativeai."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-1.5-flash",
    ) -> None:
        self.api_key = (api_key or "").strip()
        self.model = model
        self._genai = None
        self._model_obj = None

    @property
    def provider_name(self) -> str:
        return "Google Gemini"

    @property
    def model_name(self) -> str:
        return self.model

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def _init_client(self):
        if not self.is_configured():
            raise LLMError(
                "Gemini API Key is not configured. Please supply a key in the sidebar or via the GEMINI_API_KEY environment variable.",
                provider=self.provider_name,
            )

        if self._model_obj is None:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._genai = genai
                self._model_obj = genai.GenerativeModel(self.model)
            except Exception as e:
                raise LLMError(f"Failed to initialize Gemini client: {e}", provider=self.provider_name) from e

        return self._model_obj

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
    ) -> LLMResponse:
        """Generates response using Gemini SDK with 1-time retry on transient errors."""
        self._init_client()
        model_obj = self._genai.GenerativeModel(self.model)

        full_prompt = prompt
        if system_prompt:
            full_prompt = f"System Instructions:\n{system_prompt}\n\nUser Task:\n{prompt}"

        generation_config = {
            "temperature": temperature,
            "max_output_tokens": 1024,
        }

        max_attempts = 2
        last_exception = None

        for attempt in range(1, max_attempts + 1):
            try:
                response = model_obj.generate_content(
                    full_prompt,
                    generation_config=generation_config,
                )

                raw_text = response.text if response and hasattr(response, "text") else ""

                # Strip <think>...</think> blocks if present
                import re
                clean_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
                if not clean_text:
                    clean_text = raw_text.strip()

                return LLMResponse(
                    text=clean_text,
                    provider=self.provider_name,
                    model=self.model,
                    tokens_used=None,
                )

            except Exception as exc:
                last_exception = exc
                err_msg = str(exc).lower()
                is_rate_limit = "resourceexhausted" in err_msg or "429" in err_msg or "quota" in err_msg
                is_transient = is_rate_limit or "connection" in err_msg or "timeout" in err_msg or "503" in err_msg

                if attempt < max_attempts and is_transient:
                    # 1-time backoff retry
                    time.sleep(2.0)
                    continue
                else:
                    clean_err = f"Gemini Error ({self.model}): {str(exc)}"
                    if is_rate_limit:
                        clean_err = (
                            f"Gemini Quota / Rate Limit Exceeded on {self.model} (HTTP 429). "
                            "The system attempted an automatic retry, but the rate limit remains active."
                        )
                    raise LLMError(clean_err, provider=self.provider_name, is_rate_limit=is_rate_limit) from exc

        raise LLMError(f"Gemini generation failed on {self.model}: {last_exception}", provider=self.provider_name)
