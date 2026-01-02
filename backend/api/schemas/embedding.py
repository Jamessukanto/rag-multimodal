"""
Pydantic models for embedding endpoints
"""

from pydantic import BaseModel
from typing import List, Dict, Any, Optional


class GenerateEmbeddingsRequest(BaseModel):
    """Request to generate embeddings"""
    chunk_ids: List[str]
    generate_multi: bool = True


class EmbeddingStatusResponse(BaseModel):
    """Embedding job status"""
    job_id: str
    status: str  # pending, processing, completed, failed
    progress: float  # 0.0 to 1.0
    num_embeddings: Optional[int] = None


class EmbedQueryRequest(BaseModel):
    """Request to embed a query"""
    query: str


class EmbedQueryResponse(BaseModel):
    """Response with query embedding"""
    query: str
    embedding: List[float]
    dimension: int

