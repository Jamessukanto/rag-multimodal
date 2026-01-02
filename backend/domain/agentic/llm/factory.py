"""
Factory for creating LLM clients
"""

import os
from typing import Optional

from domain.agentic.llm.base import BaseLLMClient
from domain.agentic.llm.groq_client import GroqClient
from core.config import settings
from core.exceptions import LLMError

# Optional imports for other providers
try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


def create_llm_client(provider: Optional[str] = None, api_key: Optional[str] = None) -> BaseLLMClient:
    """
    Create an LLM client based on configuration.
    
    Args:
        provider: LLM provider name (overrides settings)
        api_key: API key (overrides environment variable)
        
    Returns:
        BaseLLMClient instance
        
    Raises:
        LLMError: If provider is not supported or client creation fails
    """
    provider = (provider or settings.llm_provider).lower()
    
    if provider == "groq":
        api_key = api_key or os.getenv("GROQ_API_KEY")
        return GroqClient(
            model=settings.llm_model,
            max_tokens=settings.llm_max_tokens,
            api_key=api_key,
        )
    elif provider == "anthropic":
        if not ANTHROPIC_AVAILABLE:
            raise LLMError("anthropic package not installed")
        # TODO: Implement AnthropicClient 
        raise LLMError("Anthropic client not yet implemented")
    elif provider == "openai":
        if not OPENAI_AVAILABLE:
            raise LLMError("openai package not installed")
        # TODO: Implement AnthropicClient 
        raise LLMError("Anthropic client not yet implemented")
    else:
        raise LLMError(f"Unknown LLM provider: {provider}")

