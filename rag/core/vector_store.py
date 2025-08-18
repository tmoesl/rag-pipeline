"""Vector store implementation using PostgreSQL with pgvector."""

import json
from typing import Any

import psycopg
from pgvector.psycopg import register_vector
from psycopg import sql
from psycopg.rows import dict_row

from rag.config.settings import get_settings
from rag.core import RAGError


class VectorStore:
    """Handles vector storage and similarity search using PostgreSQL with pgvector."""

    def __init__(self):
        """Initialize vector store with database connection."""
        settings = get_settings()
        self.schema = settings.database.schema_name
        self.conn: psycopg.Connection = self._establish_connection()

    def _establish_connection(self) -> psycopg.Connection:
        """Establish connection to PostgreSQL database."""
        try:
            settings = get_settings()
            conn_str = settings.database.conn_string
            conn = psycopg.connect(conn_str)
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            register_vector(conn)
            return conn
        except Exception as e:
            raise RAGError(f"Database connection failed: {e}") from e

    def add_document_with_chunks(
        self,
        title: str,
        source: str,
        chunks: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """
        Add a document with its chunks to the vector store.

        Args:
            title: Document title
            source: Document source/URL
            chunks: List of chunks with content, embedding, and metadata
            metadata: Optional metadata for the document

        Returns:
            Document ID

        Raises:
            RAGError: If database insertion fails
        """
        try:
            with self.conn.cursor() as cur:
                # Insert document
                result = cur.execute(
                    sql.SQL(
                        """
                        INSERT INTO {}.documents (title, source, metadata) 
                        VALUES (%s, %s, %s) 
                        RETURNING id
                        """
                    ).format(sql.Identifier(self.schema)),
                    (title, source, json.dumps(metadata or {})),
                )
                document_id = result.fetchone()[0]  # type: ignore

                # Insert chunks
                for chunk in chunks:
                    cur.execute(
                        sql.SQL(
                            """
                            INSERT INTO {}.chunks 
                            (document_id, content, chunk_index, metadata, embedding)
                            VALUES (%s, %s, %s, %s, %s)
                            """
                        ).format(sql.Identifier(self.schema)),
                        (
                            document_id,
                            chunk["content"],
                            chunk["metadata"]["chunk_index"],
                            json.dumps(chunk["metadata"]),
                            chunk["embedding"],
                        ),
                    )

                self.conn.commit()
                return document_id
        except Exception as e:
            raise RAGError(f"Database insertion failed: {e}") from e

    def get_existing_sources(self, sources: list[str]) -> set[str]:
        """
        Check which sources already exist in the database.

        Args:
            sources: List of source paths/URLs to check

        Returns:
            Set of sources that already exist
        """
        if not sources:
            return set()

        with self.conn.cursor() as cur:
            result = cur.execute(
                sql.SQL("SELECT source FROM {}.documents WHERE source = ANY(%s)").format(
                    sql.Identifier(self.schema)
                ),
                (sources,),
            )
            return {row[0] for row in result.fetchall()}

    def semantic_search(
        self,
        query_embedding: list[float],
        k: int = 5,
        threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Perform semantic search using embedding similarity (inner product).

        Args:
            query_embedding: Query vector
            k: Number of results to return
            threshold: Minimum similarity threshold

        Returns:
            List of semantically similar chunks with metadata

        Raises:
            RAGError: If the database search fails
        """
        try:
            with self.conn.cursor(row_factory=dict_row) as cur:
                # Base query with similarity score using inner product
                base_query = sql.SQL("""
                    SELECT
                        c.id,
                        c.content,
                        c.chunk_index,
                        c.metadata,
                        d.title,
                        d.source,
                        d.metadata as doc_metadata,
                        -(c.embedding <#> %s::vector) as similarity,
                        c.created_at
                    FROM {}.chunks c
                    JOIN {}.documents d ON c.document_id = d.id
                """).format(sql.Identifier(self.schema), sql.Identifier(self.schema))

                # Add threshold filter if specified
                if threshold is not None:
                    where_clause = sql.SQL(
                        " WHERE -(c.embedding <#> %s::vector) >= "
                    ) + sql.Literal(threshold)
                    order_clause = sql.SQL(" ORDER BY c.embedding <#> %s::vector LIMIT %s")
                    query = base_query + where_clause + order_clause
                    result = cur.execute(
                        query, (query_embedding, query_embedding, query_embedding, k)
                    )
                else:
                    query = base_query + sql.SQL(" ORDER BY c.embedding <#> %s::vector LIMIT %s")
                    result = cur.execute(query, (query_embedding, query_embedding, k))

                # Extract results (rows) as dicts
                results = [dict(r) for r in result.fetchall()]

                # Merge doc_metadata into metadata for easier access, remove doc_metadata
                for row in results:
                    row["metadata"] = (row.get("metadata") or {}) | (row.get("doc_metadata") or {})
                    row.pop("doc_metadata", None)

                return results

        except Exception as e:
            raise RAGError(f"Semantic search failed: {e}") from e

    def keyword_search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """
        Perform keyword search using PostgreSQL full-text search.

        Args:
            query: Search query text
            k: Number of results to return

        Returns:
            List of matching chunks with metadata and text search rank

        Raises:
            RAGError: If the database search fails
        """
        try:
            with self.conn.cursor(row_factory=dict_row) as cur:
                query_sql = sql.SQL("""
                    SELECT
                        c.id,
                        c.content,
                        c.chunk_index,
                        c.metadata,
                        d.title,
                        d.source,
                        d.metadata as doc_metadata,
                        ts_rank(c.fts, plainto_tsquery('english', %s)) as rank,
                        c.created_at
                    FROM {}.chunks c
                    JOIN {}.documents d ON c.document_id = d.id
                    WHERE c.fts @@ plainto_tsquery('english', %s)
                    ORDER BY ts_rank(c.fts, plainto_tsquery('english', %s)) DESC
                    LIMIT %s
                """).format(sql.Identifier(self.schema), sql.Identifier(self.schema))

                result = cur.execute(query_sql, (query, query, query, k))

                # Extract results (rows) as dicts
                results = [dict(r) for r in result.fetchall()]

                # Merge doc_metadata into metadata for easier access, remove doc_metadata
                for row in results:
                    row["metadata"] = (row.get("metadata") or {}) | (row.get("doc_metadata") or {})
                    row.pop("doc_metadata", None)

                return results
        except Exception as e:
            raise RAGError(f"Keyword search failed: {e}") from e

    def get_document_count(self) -> int:
        """Get the total number of documents in the store."""
        with self.conn.cursor() as cur:
            result = cur.execute(
                sql.SQL("SELECT COUNT(*) FROM {}.documents").format(sql.Identifier(self.schema))
            )
            return result.fetchone()[0]  # type: ignore

    def get_chunk_count(self) -> int:
        """Get the total number of chunks in the store."""
        with self.conn.cursor() as cur:
            result = cur.execute(
                sql.SQL("SELECT COUNT(*) FROM {}.chunks").format(sql.Identifier(self.schema))
            )
            return result.fetchone()[0]  # type: ignore

    def clear_database(self):
        """Clear all data from the vector database."""
        with self.conn.cursor() as cur:
            cur.execute(
                sql.SQL("TRUNCATE TABLE {}.documents RESTART IDENTITY CASCADE").format(
                    sql.Identifier(self.schema)
                )
            )
            self.conn.commit()

    def get_stats(self) -> dict[str, Any]:
        """Get database statistics."""
        settings = get_settings()
        return {
            "total_documents": self.get_document_count(),
            "total_chunks": self.get_chunk_count(),
            "embedding_dimensions": settings.embedding_dimensions,
            "embedding_max_tokens": settings.embedding_max_tokens,
            "keyword_search_enabled": settings.keyword_search_enabled,
        }

    def close(self):
        """Close the database connection."""
        self.conn.close()
