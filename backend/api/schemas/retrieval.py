"""
Pydantic models for retrieval endpoints
"""

from pydantic import BaseModel
from typing import List, Dict, Any, Optional


class SearchRequest(BaseModel):
    """Request for semantic search"""
    query: str
    top_k_ann: Optional[int] = None
    top_k_rerank: Optional[int] = None
    filter: Optional[Dict[str, Any]] = None


class SearchResult(BaseModel):
    """Single search result"""
    chunk_id: str
    score: float
    metadata: Dict[str, Any]


class SearchResponse(BaseModel):
    """Response from search"""
    query: str
    results: List[SearchResult]
    num_results: int


class AnnOnlyRequest(BaseModel):
    """Request for ANN-only search"""
    query: str
    top_k: Optional[int] = None
    filter: Optional[Dict[str, Any]] = None

