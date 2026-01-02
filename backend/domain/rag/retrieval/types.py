"""
Retrieval data types
"""

from typing import Dict, Any
from pydantic import BaseModel


class RetrievalResult(BaseModel):
    """
    Standardized retrieval result.
    
    All retrieval methods (ANNRetriever.retrieve, Reranker.rerank, RetrievalService.search)
    return List[RetrievalResult] (or List[Dict] that matches this structure).
    """
    chunk_id: str
    score: float
    metadata: Dict[str, Any]  # Includes 'chunk_name' and other chunk/document fields

