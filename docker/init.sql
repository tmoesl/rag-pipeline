-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create schema for RAG pipeline
CREATE SCHEMA IF NOT EXISTS rag;

-- Create documents table for storing document metadata
CREATE TABLE IF NOT EXISTS rag.documents (
    id BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    title TEXT NOT NULL,
    source TEXT NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create chunks table for storing document chunks
CREATE TABLE IF NOT EXISTS rag.chunks (
    id BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    document_id BIGINT NOT NULL REFERENCES rag.documents(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    metadata JSONB,
    embedding vector(1536), -- OpenAI embeddings dimension
    fts TSVECTOR GENERATED ALWAYS AS (
        to_tsvector('english', content)
    ) STORED,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create index on embeddings for fast similarity search with dot product (for normalized OpenAI embeddings)
-- https://supabase.com/docs/guides/ai/vector-indexes/hnsw-indexes
CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON rag.chunks USING hnsw (embedding vector_ip_ops);

-- Full-text search indexes
CREATE INDEX IF NOT EXISTS chunks_fts_idx ON rag.chunks USING GIN (fts);

-- Create GIN indexes on metadata for efficient JSON queries
CREATE INDEX IF NOT EXISTS documents_metadata_idx ON rag.documents USING GIN (metadata);
CREATE INDEX IF NOT EXISTS chunks_metadata_idx ON rag.chunks USING GIN (metadata);

-- Add function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_modified_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create triggers to automatically update the updated_at column
CREATE TRIGGER update_documents_modtime
BEFORE UPDATE ON rag.documents
FOR EACH ROW EXECUTE FUNCTION update_modified_column();

CREATE TRIGGER update_chunks_modtime
BEFORE UPDATE ON rag.chunks
FOR EACH ROW EXECUTE FUNCTION update_modified_column(); 