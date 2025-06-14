# src/configs/settings.py

from pydantic_settings import BaseSettings
from pydantic import AnyUrl, Field, ConfigDict

class Settings(BaseSettings):
    # LLM (Ollama)
    LLM_BASE_URL: AnyUrl = Field("http://localhost:11434", env="OLLAMA_BASE_URL")
    LLM_MODEL_NAME: str = Field("llama3.1:8b", env="LLAMA_MODEL_NAME")
    LLM_TIMEOUT: float = Field(30.0, env="LLM_TIMEOUT")
    LLM_RETRY_ATTEMPTS: int = Field(2, env="LLM_RETRY_ATTEMPTS")

    # Redis (для кэша, аналитики и т.п.)
    REDIS_URL: str = Field("redis://redis:6379/0", env="REDIS_URL")

    # Vector store backend: 'chroma' | 'qdrant' | 'mcp'
    VECTOR_BACKEND: str = Field("qdrant", env="VECTOR_BACKEND")
    CHROMA_PERSIST_DIR: str = Field("chroma_db", env="CHROMA_PERSIST_DIR")
    CHROMA_SERVER_HOST: str | None = Field(None, env="CHROMA_SERVER_HOST")
    CHROMA_SERVER_PORT: int | None = Field(None, env="CHROMA_SERVER_PORT")
    EMBEDDING_MODEL_NAME: str = Field("sentence-transformers/all-MiniLM-L6-v2", env="EMBEDDING_MODEL_NAME")
    QDRANT_URL: AnyUrl = Field("http://qdrant:6333", env="QDRANT_URL")

    # MCP
    MCP_BASE_URL: AnyUrl = Field("http://mcp-server:8000", env="MCP_BASE_URL")

    # Workflow
    MAX_ITERATIONS: int = Field(3, env="WORKFLOW_MAX_ITER")
    SKIP_QUALITY_CHECK: bool = Field(False, env="WORKFLOW_SKIP_QUALITY")
    ENABLE_CONTEXT_ENRICHMENT: bool = Field(True, env="WORKFLOW_CONTEXT_ENRICH")
    PARALLEL_PROCESSING: bool = Field(False, env="WORKFLOW_PARALLEL")

    # TTL для кэша
    TOPIC_CACHE_TTL: int = Field(3600, env="TOPIC_CACHE_TTL")
    DOC_CLASS_CACHE_TTL: int = Field(86400, env="DOC_CLASS_CACHE_TTL")

    # Разрешить игнорировать лишние переменные окружения
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
