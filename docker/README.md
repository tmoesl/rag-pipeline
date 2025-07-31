# PostgreSQL Vector Database Setup for RAG Pipeline

This directory contains configuration for a PostgreSQL database with pgvector extension to support vector similarity search for RAG (Retrieval Augmented Generation) applications.

## Quick Start

```bash
# Start the database
cd docker
docker compose up -d

# Verify it's running
docker compose ps

# Connect and test
docker compose exec pgvector psql -U postgres -d vectordb

# Shut down the database and clean volumes
docker compose down [-v]
```

## Schema Design

The database uses a dedicated `rag` schema with two main tables:

### Documents Table

Stores metadata about each source document:

- `id`: Auto-incrementing identity primary key
- `title`: Document title
- `source`: Source location/identifier
- `metadata`: JSONB field for flexible document attributes
- `created_at`, `updated_at`: Timestamps for tracking creation and updates

### Chunks Table

Stores actual document chunks with embeddings:

- `id`: Auto-incrementing identity primary key
- `document_id`: Foreign key reference to parent document
- `content`: The actual text content of the chunk
- `chunk_index`: Sequential ordering within parent document
- `metadata`: JSONB field for chunk-specific attributes
- `embedding`: High-dimensional vector for OpenAI embeddings (dimensions rely on embedding model)
- `fts`: Full-text search vector for text search
- `created_at`, `updated_at`: Timestamps for tracking creation and updates

## Search Capabilities

### Vector Similarity Search

The schema uses HNSW (Hierarchical Navigable Small World) indexes for efficient approximate nearest neighbor search:

```sql
CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON rag.chunks USING hnsw (embedding vector_ip_ops);
```

- `vector_ip_ops`: Uses dot product similarity, optimized for normalized embeddings like those from OpenAI
- HNSW provides logarithmic search time complexity instead of linear
- Supplement information can be found here: [Supabase (HNSW)](https://supabase.com/docs/guides/ai/vector-indexes/hnsw-indexes)

### Full-Text Search

The schema integrates PostgreSQL's full-text search capabilities:

- Generated TSVECTOR columns for both documents and chunks
- GIN indexes for efficient text search operations
- Enables hybrid search combining vector and keyword search

### Metadata Filtering

JSONB metadata with GIN indexes allows for efficient filtering:

```sql
CREATE INDEX IF NOT EXISTS documents_metadata_idx ON rag.documents USING GIN (metadata);
```

## Data Management

- Foreign key constraints ensure referential integrity
- Cascading deletes maintain data consistency
- Automatic timestamp updates via triggers
- JSONB type provides flexible schema evolution

## Docker Setup

The database runs in Docker with:

- PostgreSQL 17 with pgvector extension
- Persistent volume storage
- Health checks
- Resource limits
- Automatic schema initialization (`init.sql`)