"""Generation pipeline that orchestrates prompt formatting and LLM interaction."""

from typing import Any

from app.src.core.llm_service import LLMService


class GenerationPipeline:
    """Orchestrates the complete generation pipeline using existing components."""

    def __init__(self):
        """Initialize the pipeline with LLM service."""
        self.llm_service = LLMService()

    def generate_response(self, query: str, context: list[dict[str, Any]]) -> str:
        """
        Generate a text response using the retrieved context.

        Args:
            query: User's question
            context: Retrieved context chunks

        Returns:
            Text response to the user's question
        """
        print("Generating response...")
        messages = self._format_prompt(query, context)
        return self.llm_service.generate_response(messages)

    def _format_prompt(self, query: str, context: list[dict[str, Any]]) -> list[dict[str, str]]:
        """
        Format the prompt with context and query.

        Args:
            query: User's question
            context: Retrieved context chunks

        Returns:
            Formatted messages for LLM
        """
        # Prepare context for prompt
        context_texts = []
        for i, doc in enumerate(context, 1):
            chunk_info = f"Chunk {doc.get('chunk_index', i)} from '{doc.get('title', 'Unknown')}"
            context_texts.append(
                f"[{chunk_info}] (Similarity: {doc['similarity']:.3f})\n{doc['content']}"
            )

        combined_context = "\n\n---\n\n".join(context_texts)

        # Create system prompt
        system_prompt = """You are a helpful AI assistant that answers questions based on the provided context. 
        Use the context to answer the user's question accurately and comprehensively. 
        If the context doesn't contain enough information to fully answer the question, say so.
        Always cite which document(s) and chunks you're using for your answer."""

        # Create messages
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Context:\n{combined_context}\n\nQuestion: {query}",
            },
        ]

        return messages

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the generation pipeline."""
        return self.llm_service.get_stats()
