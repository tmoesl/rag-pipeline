"""Main RAG system that orchestrates the entire pipeline."""

import warnings
from typing import Any

from app.src.config.settings import SIMILARITY_THRESHOLD, TOP_K_RESULTS
from app.src.core.document_processor import DocumentProcessor
from app.src.core.embedding_service import EmbeddingService
from app.src.core.vector_store import VectorStore
from app.src.pipelines.generation import GenerationPipeline
from app.src.pipelines.retrieval import RetrievalPipeline

warnings.filterwarnings("ignore", message="'pin_memory' argument is set as true")


class RAGSystem:
    """Main RAG system that orchestrates document ingestion, retrieval, and generation."""

    def __init__(self):
        """Initialize the RAG system with all pipeline components."""
        # Create single VectorStore instance as source of truth
        self.vector_store = VectorStore()

        # Core services
        self.document_processor = DocumentProcessor()
        self.embedding_service = EmbeddingService()

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
        # Step 1: Document extraction and chunking
        chunks = self.document_processor.process_document(source, title)

        # Step 2: Generate embeddings (batch with progress)
        chunks_with_embeddings = self.embedding_service.add_embeddings_to_chunks(chunks)

        # Step 3: Store in vector database
        document_title = title or chunks[0]["metadata"]["title"]
        self.vector_store.add_document_with_chunks(
            title=document_title,
            source=source,
            chunks=chunks_with_embeddings,
            metadata=metadata,
        )

        # Print pipeline statistics
        stats = self.document_processor.get_chunk_stats(chunks)
        print(f"\nTotal chunks: {stats['total_chunks']}")
        print(f"Average tokens per chunk: {stats['avg_tokens_per_chunk']:.1f}\n")

    def ingest_multiple_documents(self, sources: list[dict[str, Any]]) -> None:
        """
        Ingest multiple documents in batch.

        Args:
            sources: List of dicts with 'source', optional 'title', and optional 'metadata' keys
        """
        for i, source_info in enumerate(sources, 1):
            source = source_info["source"]
            doc_title = source_info.get("title")
            doc_metadata = source_info.get("metadata")

            print(f"\nProcessing document {i}/{len(sources)}: {source}")

            try:
                self.ingest_document(source=source, title=doc_title, metadata=doc_metadata)
            except Exception as e:
                print(f"Error processing document {source}: {e}")
                continue

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
        # Vector/database stats
        vector_stats = self.vector_store.get_stats()

        # Embedding/LLM stats
        generation_stats = self.generation_pipeline.get_stats()

        return {
            **vector_stats,
            "embedding_model": self.embedding_service.model,
            "embedding_dimensions": self.embedding_service.dimensions,
            **generation_stats,
        }

    def clear_database(self):
        """Clear all documents and chunks from the database."""
        self.vector_store.clear_database()

    def close(self):
        """Close database connections and cleanup resources."""
        self.vector_store.close()
