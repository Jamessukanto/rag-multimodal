"""
Request/Response schemas for document management
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from datetime import datetime


class DocumentUploadResponse(BaseModel):
    """Response after uploading a document"""
    doc_id: str
    status: str
    doc_name: str
    doc_size: int
    upload_date: Optional[str] = None


class DocumentUploadsResponse(BaseModel):
    """Response after uploading multiple documents"""
    documents: List[DocumentUploadResponse]
    total: int
    successful: int
    failed: int


class DocumentStatusResponse(BaseModel):
    """Response for document status"""
    doc_id: str
    status: str
    num_chunks: int
    doc_name: str
    doc_size: int
    upload_date: Optional[str] = None


class DocumentInfo(BaseModel):
    """Document information"""
    doc_id: str
    doc_name: str
    doc_size: int
    upload_date: Optional[str] = None
    status: str
    doc_authors: Optional[str] = None
    doc_abstract: Optional[str] = None
    doc_path: Optional[str] = None
    doc_published: Optional[str] = None
    num_chunks: Optional[int] = None  # Present in list_documents response
    chunks: Optional[List[Dict[str, Any]]] = None  # Present in get_document_with_chunks response


class DocumentListResponse(BaseModel):
    """Response for listing documents"""
    documents: List[DocumentInfo]
    total: int


class DocumentDeleteResponse(BaseModel):
    """Response after deleting a document"""
    doc_id: str
    status: str


class DocumentDeleteAllResponse(BaseModel):
    """Response after deleting all documents"""
    num_documents_deleted: int
    status: str

