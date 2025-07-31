"""Data ingestion pipeline that orchestrates document processing, embedding, and storage."""

from typing import Any

from app.src.core.document_processor import DocumentProcessor
from app.src.core.embedding_service import EmbeddingService
from app.src.core.vector_store import VectorStore


class DataIngestionPipeline:
    """Orchestrates the complete data ingestion pipeline using existing components."""

    def __init__(self, vector_store: VectorStore):
        """Initialize the pipeline with all required components."""
        self.document_processor = DocumentProcessor()
        self.embedding_service = EmbeddingService()
        self.vector_store = vector_store

    def process_document(
        self, source: str, title: str | None = None, metadata: dict[str, Any] | None = None
    ) -> None:
        """
        Process a single document through the complete ingestion pipeline.

        Args:
            source: Document source (URL or file path)
            title: Optional document title
            metadata: Optional document-level metadata
        """
        # Step 1: Document extraction and chunking
        chunks = self.document_processor.process_document(source, title)

        # Step 2: Generate embeddings
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

    def process_batch(self, sources: list[dict[str, Any]]) -> None:
        """
        Process multiple documents through the ingestion pipeline.

        Args:
            sources: List of dicts with 'source', optional 'title', and optional 'metadata' keys
        """
        for i, source_info in enumerate(sources, 1):
            source = source_info["source"]
            title = source_info.get("title")
            metadata = source_info.get("metadata")

            print(f"\nProcessing document {i}/{len(sources)}: {source}")

            try:
                self.process_document(source, title, metadata)
            except Exception as e:
                print(f"Error processing document {source}: {e}")
                continue

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the pipeline and stored data."""
        vector_stats = self.vector_store.get_stats()

        return {
            "total_documents": vector_stats["total_documents"],
            "total_chunks": vector_stats["total_chunks"],
            "embedding_model": self.embedding_service.model,
            "embedding_dimensions": self.embedding_service.dimensions,
            "embedding_max_tokens": vector_stats["embedding_max_tokens"],
        }
