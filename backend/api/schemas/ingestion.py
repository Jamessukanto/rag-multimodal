"""
Request/Response schemas for ingestion
"""

from pydantic import BaseModel
from typing import Optional, List, Dict, Any


class IngestAllResponse(BaseModel):
    """Response after ingesting all unprocessed documents"""
    num_documents_just_processed: int
    num_chunks_just_processed: int
    num_documents_failed: int = 0
    failed_documents: List[Dict[str, str]] = []  # List of {doc_id, error} dicts


class IngestDocumentRequest(BaseModel):
    """Request to ingest a specific document"""
    doc_id: str


class IngestDocumentResponse(BaseModel):
    """Response after ingesting a specific document"""
    doc_id: str
    status: str
    num_chunks: int
    num_embeddings: int

