"""Document processor for extracting and chunking documents."""

from typing import Any

import tiktoken
from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer

from app.src.config.settings import EMBEDDING_MAX_TOKENS, EMBEDDING_MODEL


class DocumentProcessor:
    """Handles document extraction, processing, and chunking using Docling."""

    def __init__(self):
        """Initialize the document processor with tokenizer and chunker."""
        # Set up tokenizer for chunking
        tiktoken_encoder = tiktoken.encoding_for_model(EMBEDDING_MODEL)  # cl100k_base
        self.tokenizer = OpenAITokenizer(
            tokenizer=tiktoken_encoder, max_tokens=EMBEDDING_MAX_TOKENS
        )
        self.chunker = HybridChunker(tokenizer=self.tokenizer)
        self.converter = DocumentConverter()

    def process_document(
        self, source: str, document_title: str | None = None
    ) -> list[dict[str, Any]]:
        """
        Process a document from URL or file path, returning a list of chunks with metadata.

        Args:
            source: URL or file path to the document
            document_title: Optional title for the document

        Returns:
            List of chunks with metadata
        """
        # Convert document
        conversion_result = self.converter.convert(source)
        doc = conversion_result.document

        # Use provided title or extract from document
        title = document_title or getattr(doc, "title", source.split("/")[-1])

        # Create chunks
        chunks = list(self.chunker.chunk(dl_doc=doc))

        # Process chunks with contextualization
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            # Get contextualized text (includes headings/context)
            contextualized_text = self.chunker.contextualize(chunk=chunk)

            # Extract page numbers from chunk metadata with error handling
            page_numbers = self._extract_page_numbers(chunk)

            # Extract headings from chunk metadata with error handling
            headings = self._extract_headings(chunk)

            # Create metadata
            metadata = {
                "chunk_index": i,
                "total_chunks": len(chunks),
                "source": source,
                "title": title,
                "page_numbers": page_numbers,
                "headings": headings,
            }

            processed_chunks.append({"content": contextualized_text, "metadata": metadata})

        return processed_chunks

    def _extract_page_numbers(self, chunk) -> list[int]:
        """Extract page numbers from chunk metadata."""
        return sorted(
            {
                prov.page_no
                for item in chunk.meta.doc_items
                for prov in getattr(item, "prov", [])
                if getattr(prov, "page_no", None) is not None
            }
        )

    def _extract_headings(self, chunk) -> list[str]:
        """Extract headings from chunk metadata."""
        return getattr(chunk.meta, "headings", []) or []

    def get_chunk_stats(self, chunks: list[dict[str, Any]]) -> dict[str, Any]:
        """Get statistics about the processed chunks."""
        total_tokens = 0

        for chunk in chunks:
            tokens = self.tokenizer.tokenizer.encode(chunk["content"])
            total_tokens += len(tokens)

        return {
            "total_chunks": len(chunks),
            "total_tokens": total_tokens,
            "avg_tokens_per_chunk": total_tokens / len(chunks) if chunks else 0,
        }
