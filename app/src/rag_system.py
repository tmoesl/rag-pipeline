"""Main RAG system that orchestrates the entire pipeline."""

import warnings
from typing import Any

from app.src.config.settings import (
    SIMILARITY_THRESHOLD,
    TOP_K_RESULTS,
)
from app.src.core.vector_store import VectorStore
from app.src.pipelines.generation import GenerationPipeline
from app.src.pipelines.ingestion import DataIngestionPipeline
from app.src.pipelines.retrieval import RetrievalPipeline

warnings.filterwarnings("ignore", message="'pin_memory' argument is set as true")


class RAGSystem:
    """Main RAG system that orchestrates document ingestion, retrieval, and generation."""

    def __init__(self):
        """Initialize the RAG system with all pipeline components."""
        # Create single VectorStore instance as source of truth
        self.vector_store = VectorStore()

        # Pass the shared VectorStore to pipelines
        self.data_pipeline = DataIngestionPipeline(vector_store=self.vector_store)
        self.retrieval_pipeline = RetrievalPipeline(vector_store=self.vector_store)
        self.generation_pipeline = GenerationPipeline()

    def ingest_document(
        self, source: str, title: str | None = None, metadata: dict[str, Any] | None = None
    ) -> None:
        """
        Process and store a document in the vector database.

        Args:
            source: Document source (URL or file path)
            title: Optional document title
            metadata: Optional document-level metadata
        """
        self.data_pipeline.process_document(source, title, metadata)

    def ingest_multiple_documents(self, sources: list[dict[str, Any]]) -> None:
        """
        Ingest multiple documents in batch.

        Args:
            sources: List of dicts with 'source', optional 'title', and optional 'metadata' keys
        """
        self.data_pipeline.process_batch(sources)

    def query(
        self,
        question: str,
        k: int = TOP_K_RESULTS,
        threshold: float = SIMILARITY_THRESHOLD,
    ) -> dict[str, Any]:
        """
        Main query method that combines retrieval and generation.

        Args:
            question: The user's question
            k: Number of documents to retrieve
            threshold: Similarity threshold

        Returns:
            Dictionary containing response, context, and metadata
        """
        # Retrieve relevant context
        context = self.retrieval_pipeline.retrieve_context(question, k, threshold)

        if not context:
            print("No relevant context found")  # RAG-level decision
            return {
                "response": "I couldn't find any relevant information to answer your question.",
                "context": [],
                "context_count": 0,
            }

        print(f"Found {len(context)} relevant chunks")  # RAG-level result

        # Generate response
        response = self.generation_pipeline.generate_response(question, context)

        return {
            "response": response,
            "context": context,
            "context_count": len(context),
        }

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive statistics about the RAG system."""
        ingestion_stats = self.data_pipeline.get_stats()
        generation_stats = self.generation_pipeline.get_stats()

        # Return ALL stats from both pipelines
        return {
            **ingestion_stats,
            **generation_stats,
        }

    def clear_database(self):
        """Clear all documents and chunks from the database."""
        self.vector_store.clear_database()

    def close(self):
        """Close database connections and cleanup resources."""
        self.vector_store.close()
