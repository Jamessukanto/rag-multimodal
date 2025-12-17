"""
Configuration and settings
"""

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    server_script_path: str = "/Users/jamessukanto/Desktop/codes/exps/mcp/mcp_server_docs/main.py"
    llm_provider: str = "groq"  # Options: "anthropic", "openai", "groq"
    llm_model: str = "qwen/qwen3-32b" # Model name for groq
    llm_max_tokens: int = 1000
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Ignore extra fields from .env (like GROQ_API_KEY)


settings = Settings()

