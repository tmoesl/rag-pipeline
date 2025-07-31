"""Retrieval pipeline that orchestrates query processing and similarity search."""

from typing import Any

from app.src.core.embedding_service import EmbeddingService
from app.src.core.vector_store import VectorStore


class RetrievalPipeline:
    """Orchestrates the complete retrieval pipeline using existing components."""

    def __init__(self, vector_store: VectorStore):
        """Initialize the pipeline with all required components."""
        self.embedding_service = EmbeddingService()
        self.vector_store = vector_store

    def retrieve_context(self, query: str, k: int, threshold: float) -> list[dict[str, Any]]:
        """
        Retrieve relevant context for a query through the complete retrieval pipeline.

        Args:
            query: Search query
            k: Number of results to retrieve (required)
            threshold: Similarity threshold (required)

        Returns:
            List of relevant chunks with metadata
        """
        # Step 1: Query embedding
        query_embedding = self.embedding_service.embed_query(query)

        # Step 2: Similarity search
        print("Searching for relevant context...")
        results = self.vector_store.similarity_search(
            query_embedding=query_embedding, k=k, threshold=threshold
        )

        return results
