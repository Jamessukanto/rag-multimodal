"""
Abstract base classes for storage
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path
from core.exceptions import StorageError


class BaseSingleVectorStore(ABC):
    """Abstract base class for single vector stores"""
    
    @abstractmethod
    async def add(
        self,
        chunk_id: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a vector to the store"""
        pass
    
    @abstractmethod
    async def query(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Query the store.
        
        Returns:
            List of results with 'chunk_id', 'score', 'metadata'
        """
        pass
    
    @abstractmethod
    async def delete(self, chunk_id: str) -> None:
        """Delete a vector from the store"""
        pass
    
    @abstractmethod
    async def update(
        self,
        chunk_id: str,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Update a vector in the store"""
        pass


class BaseMultiVectorStore(ABC):
    """Abstract base class for multi-vector stores"""
    
    @abstractmethod
    async def add(
        self,
        chunk_id: str,
        embeddings: List[List[float]]
    ) -> None:
        """Add multi-vectors to the store"""
        pass
    
    @abstractmethod
    async def get(self, chunk_id: str) -> Optional[List[List[float]]]:
        """Get multi-vectors for a chunk"""
        pass
    
    @abstractmethod
    async def batch_get(
        self,
        chunk_ids: List[str]
    ) -> Dict[str, List[List[float]]]:
        """Get multi-vectors for multiple chunks"""
        pass
    
    @abstractmethod
    async def delete(self, chunk_id: str) -> None:
        """Delete multi-vectors from the store"""
        pass


class BaseDocumentSQLStore(ABC):
    """Abstract base class for document SQL stores"""
    
    @abstractmethod
    async def upsert_document(
        self,
        doc_id: str,
        doc_name: str,
        doc_size: int,
        upload_date: Any = None,
        status: str = "uploaded",
        doc_authors: str = None,
        doc_abstract: str = None,
        doc_path: str = None,
        doc_published: Any = None
    ) -> None:
        """Upsert document metadata"""
        pass
    
    @abstractmethod
    async def upsert_chunk(
        self,
        chunk_id: str,
        doc_id: str,
        chunk_path: str = None,
        chunk_name: str = None,
        chunk_source: str = None,
        chunk_level: str = None
    ) -> None:
        """Upsert chunk metadata"""
        pass
    
    @abstractmethod
    async def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get document metadata"""
        pass
    
    @abstractmethod
    async def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get chunk metadata"""
        pass
    
    @abstractmethod
    async def get_chunks_by_document(self, doc_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a document"""
        pass
    
    @abstractmethod
    async def get_document_by_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get document that contains a specific chunk"""
        pass
    
    @abstractmethod
    async def delete_document(self, doc_id: str) -> None:
        """Delete document and all its chunks"""
        pass
    
    @abstractmethod
    async def list_documents(self, filter: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """List documents with optional status filter."""
        pass
    
    @abstractmethod
    async def get_document_with_chunks(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get document with all its chunks"""
        pass
    
    @abstractmethod
    async def update_document_status(self, doc_id: str, status: str) -> None:
        """Update document status"""
        pass
    
    @abstractmethod
    async def delete_chunk(self, chunk_id: str) -> None:
        """Delete chunk metadata"""
        pass


class BaseFileStore(ABC):
    """Abstract base class for file stores"""
    
    @abstractmethod
    async def save_file(self, doc_id: str, file_content_bytes: bytes) -> Path:
        """Save PDF file"""
        pass
    
    @abstractmethod
    async def get_file(self, doc_id: str) -> Optional[bytes]:
        """Get PDF file content"""
        pass
    
    @abstractmethod
    async def delete_file(self, doc_id: str) -> None:
        """Delete PDF file"""
        pass
    
    @abstractmethod
    def file_exists(self, doc_id: str) -> bool:
        """Check if file exists"""
        pass
    
    @abstractmethod
    def get_file_path(self, doc_id: str) -> Path:
        """Get file path for a document ID"""
        pass

