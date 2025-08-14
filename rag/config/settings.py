"""Configuration settings for the RAG pipeline."""

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseConfig(BaseSettings):
    """Database configuration for PostgreSQL with pgvector."""

    # Database connection
    host: str = Field(alias="DB_HOST", default="localhost")
    port: int = Field(alias="DB_PORT", default=5432)
    name: str = Field(alias="DB_NAME", default="vectordb")
    user: str = Field(alias="DB_USER", default="postgres")
    password: str = Field(alias="DB_PASSWORD", default="postgres")

    # Database schema
    schema_name: str = Field(alias="SCHEMA_NAME", default="rag")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    @property
    def conn_string(self) -> str:
        """Build psycopg3 connection string."""
        return (
            f"host={self.host} "
            f"port={self.port} "
            f"dbname={self.name} "
            f"user={self.user} "
            f"password={self.password}"
        )


class Settings(BaseSettings):
    """Main RAG pipeline configuration."""

    # Database
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)

    # OpenAI configuration
    openai_api_key: SecretStr = Field(alias="OPENAI_API_KEY")
    embedding_model: str = Field(default="text-embedding-3-small", alias="EMBEDDING_MODEL")
    embedding_dimensions: int = Field(default=1536, alias="EMBEDDING_DIMENSIONS")
    chat_model: str = Field(default="gpt-4o-mini", alias="CHAT_MODEL")
    temperature: float = Field(default=0.7, alias="TEMPERATURE")
    max_output_tokens: int = Field(default=1000, alias="MAX_OUTPUT_TOKENS")
    embedding_max_tokens: int = Field(default=8191, alias="EMBEDDING_MAX_TOKENS")

    # Vector search
    top_k_results: int = Field(default=5, alias="TOP_K_RESULTS")
    semantic_threshold: float = Field(default=0.3, alias="SEMANTIC_THRESHOLD")

    # Hybrid search
    hybrid_search_alpha: float = Field(default=0.6, alias="HYBRID_SEARCH_ALPHA")
    keyword_search_enabled: bool = Field(default=True, alias="KEYWORD_SEARCH_ENABLED")

    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance (singleton)."""
    return Settings()


def get_psycopg_connection_string() -> str:
    """Build psycopg3 connection string."""
    return get_settings().database.conn_string
