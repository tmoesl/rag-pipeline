"""Reranker service using Cohere for improving document relevance."""

from typing import Any

import cohere

from rag.config.settings import get_settings
from rag.utils.logger import logger


class RerankerService:
    """Service for reranking retrieved documents using Cohere's rerank models."""

    def __init__(self):
        """Initialize the re-ranking service with Cohere."""
        settings = get_settings()
        self.client = cohere.ClientV2(api_key=settings.cohere_api_key.get_secret_value())
        self.model = settings.rerank_model

    def rerank_documents(
        self, query: str, documents: list[dict[str, Any]], top_k: int
    ) -> list[dict[str, Any]]:
        """
        Rerank documents using Cohere's rerank API.

        Args:
            query: The search query
            documents: List of document dictionaries
            top_k: Number of top documents to return after reranking

        Returns:
            List of reranked documents with added rerank_score and original_rank fields
        """
        if not documents:
            return []

        # Extract text content for reranking
        doc_texts = [doc["content"] for doc in documents]

        try:
            # Call Cohere rerank API
            reranked_response = self.client.rerank(
                model=self.model,
                query=query,
                documents=doc_texts,
                top_n=top_k,
            )

            # Build reranked results with additional metadata
            # Cohere Rerank index is the document's position in your original input list.
            reranked_docs = []
            for result in reranked_response.results:
                original_doc = documents[result.index].copy()
                original_doc["rerank_score"] = result.relevance_score
                original_doc["original_rank"] = result.index + 1
                reranked_docs.append(original_doc)

            logger.debug(f"Successfully reranked to {len(reranked_docs)} documents")
            return reranked_docs

        except Exception as e:
            logger.error(f"Reranking failed: {e}")

            # Fall back to original results (truncated to top_k)
            logger.warning("Falling back to original document order")
            return documents[:top_k]
