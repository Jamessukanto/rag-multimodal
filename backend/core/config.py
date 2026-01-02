"""
Unified configuration and settings
Combines existing LLM/MCP config with RAG config
"""

from pathlib import Path
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Load environment variables
load_dotenv()


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    Values can be set via:
    1. Environment variables (highest priority)
    2. .env file (loaded by load_dotenv())
    3. Default values below (lowest priority)
    """
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    environment: str = "development"  # development, staging, production
    log_level: str = "INFO"
    
    # ------------------------
    # Agentic
    # ------------------------

    # LLM Configuration
    llm_provider: str = "groq"  # Options: "anthropic", "openai", "groq"
    llm_model: str = "qwen/qwen3-32b"  # Model name for groq
    llm_max_tokens: int = 1000
    
    # MCP Configuration
    # For Docker: Use absolute path like "/app/mcp_server_docs/main.py"
    # For local dev: Use absolute path or relative path like "../mcp_server_docs/main.py"
    mcp_server_script_path: str = "/Users/jamessukanto/Desktop/codes/projs/rag-multimodal/mcp_server_docs/main.py"
    

    # ------------------------
    # Embedding: Jina Embedding API
    # ------------------------

    jina_api_key: str = ""
    jina_api_url: str = "https://api.jina.ai/v1/embeddings"
    jina_model: str = "jina-embeddings-v4"
    jina_timeout: int = 120
    jina_max_retries: int = 3
    jina_rate_limit: int = 10  # requests per second
    
    # ------------------------
    # Single Vector Store: ChromaDB
    # ------------------------

    vector_store_backend: str = "chromadb_embedded"  # Options: "chromadb_embedded", "chromadb_cloud"
    vector_store_collection_name: str = "embeddings"

    # Dev: Embedded 
    single_vector_store_path: Path = Path("./data/single_vector_db")
    # Prod: Cloud managed service
    chromadb_cloud_api_key: str = ""
    chromadb_cloud_tenant: str = ""
    chromadb_cloud_database: str = ""

    # ------------------------
    # Multi Vector Store: mmap np files
    # ------------------------

    # Dev:  
    multi_vector_store_path: Path = Path("./data/multi_vector_db")
    # Prod:
    # Not decided

    # ------------------------
    # Document Store: PostgreSQL
    # ------------------------
    
    # Dev:
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "postgres"
    postgres_password: str = ""
    postgres_database: str = "rag_multimodal"
    # postgresql://user:password@localhost:5432/dbname
    # Prod: Managed service connection string (e.g., Render, AWS RDS)
    postgres_db_url: str = ""
    
    # ------------------------
    # File Store
    # ------------------------
    
    # Dev:
    documents_dir: Path = Path("./data/documents")  # PDF file storage
    chunks_dir: Path = Path("./data/chunks")  # PDF page chunk storage
    
    # Prod:
    file_storage_type: str = "filesystem"  # "filesystem", "s3", or "minio"
    s3_bucket: str = ""  # For S3/MinIO storage
    s3_endpoint: str = ""  # For MinIO
    
    # ------------------------
    # RAG
    # ------------------------

    # Retrieval Parameters
    default_top_k_ann: int = 10
    default_top_k_rerank: int = 5
    
    # Ingestion
    ingestion_cache_enabled: bool = True
    max_pdf_size_mb: int = 50
    
    # Evaluation
    eval_results_dir: Path = Path("./data/eval") 
    
    class Config:
        """
        Pydantic configuration for settings loading.
        
        - env_file: Which .env file to read
        - env_file_encoding: File encoding
        - extra: What to do with extra fields in .env that aren't in this class
        """
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Ignore extra .env vars (like API keys used by libraries)


# Singleton settings instance
settings = Settings()

