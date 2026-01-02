"""
Embedding generation pipeline
"""

from domain.rag.embedding.client import JinaEmbeddingClient
from domain.rag.embedding.batch_processor import BatchProcessor
from domain.rag.embedding.types import EmbeddingResult, SingleVectorEmbedding, MultiVectorEmbedding

__all__ = [
    "JinaEmbeddingClient",
    "BatchProcessor",
    "EmbeddingResult",
    "SingleVectorEmbedding",
    "MultiVectorEmbedding",
]

