"""Embedding service for creating vector embeddings using OpenAI."""

from typing import Any

from openai import OpenAI
from tqdm import tqdm

from rag.config.settings import get_settings
from rag.core import RAGError


class EmbeddingService:
    """Handles embedding generation using OpenAI's API."""

    def __init__(self):
        """Initialize the embedding service with OpenAI client."""
        try:
            settings = get_settings()
            self.client = OpenAI(api_key=settings.openai_api_key.get_secret_value())
            self.model = settings.embedding_model
            self.dimensions = settings.embedding_dimensions
        except Exception as e:
            raise RAGError(f"Failed to initialize EmbeddingService: {e}") from e

    def create_embedding(self, text: str) -> list[float]:
        """
        Create embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            List of embedding values

        Raises:
            RAGError: If the embedding API call fails
        """
        try:
            response = self.client.embeddings.create(
                model=self.model, input=text, dimensions=self.dimensions
            )
            return response.data[0].embedding
        except Exception as e:
            raise RAGError(f"Embedding API failed with model {self.model}: {e}") from e

    def create_embeddings_batch(self, texts: list[str], batch_size: int = 100) -> list[list[float]]:
        """
        Create embeddings for multiple texts in batches.

        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process per batch

        Returns:
            List of embedding vectors

        Raises:
            RAGError: If the embedding API call fails
        """
        all_embeddings = []

        # Process in batches with progress bar
        with tqdm(total=len(texts), desc="Creating embeddings") as pbar:
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]

                try:
                    response = self.client.embeddings.create(
                        model=self.model, input=batch, dimensions=self.dimensions
                    )
                    embeddings = [item.embedding for item in response.data]
                    all_embeddings.extend(embeddings)
                except Exception as e:
                    raise RAGError(f"Embedding API failed with model {self.model}: {e}") from e

                pbar.update(len(batch))

        return all_embeddings

    def add_embeddings_to_chunks(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Add embeddings to a list of document chunks.

        Args:
            chunks: List of chunks with content and metadata

        Returns:
            List of chunks with embeddings added
        """
        # Extract texts
        texts = [chunk["content"] for chunk in chunks]

        # Create embeddings
        embeddings = self.create_embeddings_batch(texts)

        # Add embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            chunk["embedding"] = embedding

        return chunks

    def embed_query(self, query: str) -> list[float]:
        """
        Create embedding for a search query.

        Args:
            query: Search query text

        Returns:
            Query embedding vector
        """
        return self.create_embedding(query)
