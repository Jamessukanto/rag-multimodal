"""
Request/Response schemas for chunk management
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel


class ChunkInfo(BaseModel):
    """Chunk information"""
    chunk_id: str
    doc_id: str
    chunk_name: str
    chunk_path: str
    chunk_source: Optional[str] = None
    chunk_level: Optional[str] = None


class ChunkListResponse(BaseModel):
    """Response for listing chunks"""
    chunks: List[ChunkInfo]
    total: int


class ChunkDeleteResponse(BaseModel):
    """Response after deleting a chunk"""
    chunk_id: str
    status: str


class ChunkDeleteAllResponse(BaseModel):
    """Response after deleting all chunks"""
    num_chunks_deleted: int
    status: str

