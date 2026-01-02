"""
Embedding data types
"""

from typing import List, Optional
from pydantic import BaseModel


class SingleVectorEmbedding(BaseModel):
    """Single vector embedding"""
    id: str
    embedding: List[float]
    text: Optional[str] = None
    model_embed: Optional[str] = None


class MultiVectorEmbedding(BaseModel):
    """Multi-vector embedding (multiple vectors per chunk)"""
    id: str
    embeddings: List[List[float]]  # List of vectors
    text: Optional[str] = None 
    model_embed: Optional[str] = None


class EmbeddingResult(BaseModel):
    """Result from embedding generation"""
    id: str
    single_vector: Optional[SingleVectorEmbedding] = None
    multi_vectors: Optional[MultiVectorEmbedding] = None
    error: Optional[str] = None

