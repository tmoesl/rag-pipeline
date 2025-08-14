"""LLM service for text responses using OpenAI API."""

from typing import Any

from openai import OpenAI

from rag.config.settings import get_settings


class LLMService:
    """Handles LLM interactions with OpenAI API for text generation."""

    def __init__(self):
        """Initialize the LLM service with OpenAI client."""
        settings = get_settings()
        self.client = OpenAI(api_key=settings.openai_api_key.get_secret_value())
        self.model = settings.chat_model
        self.temperature = settings.temperature
        self.max_output_tokens = settings.max_output_tokens

    def generate_response(self, messages: list[dict[str, str]]) -> str:
        """
        Generate text response using OpenAI responses API.

        Args:
            messages: Formatted messages for the LLM

        Returns:
            Generated text response

        Raises:
            Exception: If the API call fails
        """
        try:
            response = self.client.responses.create(
                model=self.model,
                input=messages,  # type: ignore
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
            )
            return response.output_text
        except Exception:
            # Fallback to a basic response if API call fails
            return "I encountered an error while processing your request. Please try again."

    def get_stats(self) -> dict[str, Any]:
        """Get LLM service statistics."""
        return {
            "llm_model": self.model,
            "llm_temperature": self.temperature,
            "llm_max_output_tokens": self.max_output_tokens,
        }
