"""LLM service for text responses using OpenAI API."""

from typing import Any

from openai import OpenAI

from rag.config.settings import get_settings
from rag.core import RAGError


class LLMService:
    """Handles LLM interactions with OpenAI API for text generation."""

    def __init__(self):
        """Initialize the LLM service with OpenAI client."""
        try:
            settings = get_settings()
            self.client = OpenAI(api_key=settings.openai_api_key.get_secret_value())
            self.model = settings.chat_model
            self.temperature = settings.temperature
            self.max_output_tokens = settings.max_output_tokens
        except Exception as e:
            raise RAGError(f"Failed to initialize LLMService: {e}") from e

    def generate_response(self, messages: list[dict[str, str]]) -> str:
        """
        Generate text response using OpenAI responses API.

        Args:
            messages: Formatted messages for the LLM

        Returns:
            Generated text response

        Raises:
            RAGError: If the API call fails
        """
        try:
            response = self.client.responses.create(
                model=self.model,
                input=messages,  # type: ignore
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
            )
            return response.output_text
        except Exception as e:
            raise RAGError(f"LLM API failed with model {self.model}: {e}") from e

    def get_stats(self) -> dict[str, Any]:
        """Get LLM service statistics."""
        return {
            "llm_model": self.model,
            "llm_temperature": self.temperature,
            "llm_max_output_tokens": self.max_output_tokens,
        }
