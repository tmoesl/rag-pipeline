"""Configuration settings for the RAG pipeline."""

import os
from typing import Any

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database configuration for PostgreSQL with pgvector
DATABASE_CONFIG: dict[str, Any] = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "database": os.getenv("DB_NAME", "vectordb"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "postgres"),
}

# OpenAI configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

# Document processing configuration
EMBEDDING_MAX_TOKENS = int(os.getenv("EMBEDDING_MAX_TOKENS", "8191"))

# Vector search configuration
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))

# Hybrid search configuration
HYBRID_SEARCH_ALPHA = float(os.getenv("HYBRID_SEARCH_ALPHA", "0.6"))
KEYWORD_SEARCH_ENABLED = os.getenv("KEYWORD_SEARCH_ENABLED", "true").lower() == "true"
HYBRID_SEARCH_K_MULTIPLIER = int(os.getenv("HYBRID_SEARCH_K_MULTIPLIER", "3"))

# RAG generation configuration
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "1000"))

# Database schema configuration
SCHEMA_NAME = os.getenv("SCHEMA_NAME", "rag")


def get_psycopg_connection_string() -> str:
    """Build psycopg3 connection string."""
    config = DATABASE_CONFIG
    return (
        f"host={config['host']} "
        f"port={config['port']} "
        f"dbname={config['database']} "
        f"user={config['user']} "
        f"password={config['password']}"
    )
