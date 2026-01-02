"""
Document SQL store using PostgreSQL
"""

import logging
from typing import List, Dict, Any, Optional
from sqlalchemy import create_engine, Column, String, ForeignKey, Integer, DateTime, Enum
from sqlalchemy.orm import DeclarativeBase, sessionmaker, relationship
from datetime import datetime
import enum
from storage.base import BaseDocumentSQLStore
from core.config import settings
from core.exceptions import StorageError

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


class DocumentStatus(enum.Enum):
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    PROCESSED = "processed"
    ERROR = "error"

class ChunkSource(enum.Enum):
    PDF = "pdf"
    TEXT = "text"

class ChunkLevel(enum.Enum):
    PAGE = "page"
    PARAGRAPH = "paragraph"
    SENTENCE = "sentence"


class DocumentModel(Base):
    __tablename__ = "documents"
    
    doc_id = Column(String, primary_key=True)
    upload_date = Column(DateTime, nullable=False, default=datetime.utcnow)
    status = Column(
        Enum(DocumentStatus, name="document_status"),
        nullable=False,
        default=DocumentStatus.UPLOADED
    )
    doc_name = Column(String, nullable=False)
    doc_size = Column(Integer, nullable=False)
    doc_authors = Column(String, nullable=False)  
    doc_abstract = Column(String, nullable=False)  
    doc_path = Column(String, nullable=False)  
    doc_published = Column(DateTime, nullable=True)  # Publication date may not be available for all documents  
    
    chunks = relationship("ChunkModel", back_populates="document", cascade="all, delete-orphan")
    

class ChunkModel(Base):
    __tablename__ = "chunks"
    
    chunk_id = Column(String, primary_key=True)
    doc_id = Column(String, ForeignKey("documents.doc_id"), nullable=False)
    chunk_name = Column(String, nullable=False)  
    chunk_path = Column(String, nullable=False)  
    chunk_source = Column(
        Enum(ChunkSource, name="chunk_source"),
        nullable=False,
        default=ChunkSource.PDF
    )
    chunk_level = Column(
        Enum(ChunkLevel, name="chunk_level"),
        nullable=False,
        default=ChunkLevel.PAGE
    )

    document = relationship("DocumentModel", back_populates="chunks")


class DocumentSQLStore(BaseDocumentSQLStore):
    """Document SQL store using PostgreSQL"""

    def __init__(self):
        db_url = self._get_db_url()
        logger.info(f"Connecting to PostgreSQL database")

        self.engine = create_engine(
            db_url,
            pool_pre_ping=True,  # Verify connections before using
            pool_size=5,         # Connection pool size
            max_overflow=10      # Max connections beyond pool_size
        )

        # Create tables if they don't exist
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
    
    def _get_db_url(self) -> str:
        # Prioritize postgres_db_url for managed services like Render
        if settings.postgres_db_url:
            return settings.postgres_db_url
        
        # Build URL from individual parameters
        if not settings.postgres_password:
            raise ValueError(
                "PostgreSQL password is required. Set POSTGRES_PASSWORD environment variable "
                "or POSTGRES_DB_URL connection string."
            )
        return (
            f"postgresql://{settings.postgres_user}:{settings.postgres_password}"
            f"@{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_database}"
        )

    async def upsert_document(
        self,
        doc_id: str,
        doc_name: str,
        doc_size: int,
        upload_date: datetime = None,
        status: str = "uploaded",
        doc_authors: str = None,
        doc_abstract: str = None,
        doc_path: str = None,
        doc_published: datetime = None
    ) -> None:
        try:
            if status not in ["uploaded", "processing", "processed", "error"]:
                raise ValueError(f"Invalid status: {status}. Must be one of: uploaded, processing, processed, error")
            
            with self.SessionLocal.begin() as session:
                doc = DocumentModel(
                    doc_id=doc_id,
                    doc_name=doc_name,
                    doc_size=doc_size,
                    upload_date=upload_date or datetime.utcnow(),
                    status=DocumentStatus(status),
                    doc_authors=doc_authors,
                    doc_abstract=doc_abstract,
                    doc_path=doc_path,
                    doc_published=doc_published
                )
                session.merge(doc)  
        except Exception as e:
            logger.error(f"Error adding document {doc_id}: {e}")
            raise StorageError(f"Failed to add document: {e}")
    
    async def upsert_chunk(
        self,
        chunk_id: str,
        doc_id: str,
        chunk_path: str,
        chunk_name: str,
        chunk_source: str = None,
        chunk_level: str = None
    ) -> None:
        try:
            try:
                chunk_source_enum = ChunkSource.PDF if chunk_source is None else ChunkSource(chunk_source)
            except ValueError:
                raise ValueError(f"Invalid chunk_source: {chunk_source}. Must be one of: {[e.value for e in ChunkSource]}")
            
            try:
                chunk_level_enum = ChunkLevel.PAGE if chunk_level is None else ChunkLevel(chunk_level)
            except ValueError:
                raise ValueError(f"Invalid chunk_level: {chunk_level}. Must be one of: {[e.value for e in ChunkLevel]}")
            
            with self.SessionLocal.begin() as session:
                chunk = ChunkModel(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    chunk_path=chunk_path,
                    chunk_name=chunk_name,
                    chunk_source=chunk_source_enum,
                    chunk_level=chunk_level_enum
                )
                session.merge(chunk)
        except Exception as e:
            logger.error(f"Error adding chunk {chunk_id}: {e}")
            raise StorageError(f"Failed to add chunk: {e}")
    
    async def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        try:
            with self.SessionLocal.begin() as session:
                doc = session.query(DocumentModel).filter_by(doc_id=doc_id).first()
                if not doc:
                    return None
                return {
                    "doc_id": doc.doc_id,
                    "doc_name": doc.doc_name,
                    "doc_size": doc.doc_size,
                    "upload_date": doc.upload_date.isoformat() if doc.upload_date else None,
                    "status": doc.status.value if isinstance(doc.status, DocumentStatus) else doc.status,
                    "doc_authors": doc.doc_authors,
                    "doc_abstract": doc.doc_abstract,
                    "doc_path": doc.doc_path,
                    "doc_published": doc.doc_published.isoformat() if doc.doc_published else None,
                }
        except Exception as e:
            logger.error(f"Error getting document {doc_id}: {e}")
            raise StorageError(f"Failed to get document: {e}")
    
    async def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        try:
            with self.SessionLocal.begin() as session:
                chunk = session.query(ChunkModel).filter_by(chunk_id=chunk_id).first()
                if not chunk:
                    return None
                return {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "chunk_name": chunk.chunk_name,
                    "chunk_path": chunk.chunk_path,
                    "chunk_source": chunk.chunk_source.value if isinstance(chunk.chunk_source, ChunkSource) else chunk.chunk_source,
                    "chunk_level": chunk.chunk_level.value if isinstance(chunk.chunk_level, ChunkLevel) else chunk.chunk_level,
                }
        except Exception as e:
            logger.error(f"Error getting chunk {chunk_id}: {e}")
            raise StorageError(f"Failed to get chunk: {e}")
    
    async def get_chunks_by_document(self, doc_id: str) -> List[Dict[str, Any]]:
        try:
            with self.SessionLocal.begin() as session:
                chunks = session.query(ChunkModel).filter_by(doc_id=doc_id).all()
                return [
                    {
                        "chunk_id": chunk.chunk_id,
                        "doc_id": chunk.doc_id,
                        "chunk_name": chunk.chunk_name,
                        "chunk_path": chunk.chunk_path,
                        "chunk_source": chunk.chunk_source.value if isinstance(chunk.chunk_source, ChunkSource) else chunk.chunk_source,
                        "chunk_level": chunk.chunk_level.value if isinstance(chunk.chunk_level, ChunkLevel) else chunk.chunk_level,
                    }
                    for chunk in chunks
                ]
        except Exception as e:
            logger.error(f"Error getting chunks for document {doc_id}: {e}")
            raise StorageError(f"Failed to get chunks: {e}")
    
    async def get_document_by_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get document that contains a specific chunk"""
        try:
            with self.SessionLocal.begin() as session:
                chunk = session.query(ChunkModel).filter_by(chunk_id=chunk_id).first()
                if not chunk:
                    return None
                
                # Get document via relationship
                doc = chunk.document
                if not doc:
                    return None
                
                return {
                    "doc_id": doc.doc_id,
                    "doc_name": doc.doc_name,
                    "doc_size": doc.doc_size,
                    "upload_date": doc.upload_date.isoformat() if doc.upload_date else None,
                    "status": doc.status.value if isinstance(doc.status, DocumentStatus) else doc.status,
                    "doc_authors": doc.doc_authors,
                    "doc_abstract": doc.doc_abstract,
                    "doc_path": doc.doc_path,
                    "doc_published": doc.doc_published.isoformat() if doc.doc_published else None,
                }
        except Exception as e:
            logger.error(f"Error getting document for chunk {chunk_id}: {e}")
            raise StorageError(f"Failed to get document by chunk: {e}")
    
    async def delete_document(self, doc_id: str) -> None:
        """Delete document and all its chunks (cascade delete)"""
        try:
            with self.SessionLocal.begin() as session:
                doc = session.query(DocumentModel).filter_by(doc_id=doc_id).first()
                if doc:
                    session.delete(doc)
        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {e}")
            raise StorageError(f"Failed to delete document: {e}")
    
    async def delete_chunk(self, chunk_id: str) -> None:
        """Delete chunk metadata"""
        try:
            with self.SessionLocal.begin() as session:
                chunk = session.query(ChunkModel).filter_by(chunk_id=chunk_id).first()
                if chunk:
                    session.delete(chunk)
                else:
                    raise StorageError(f"Chunk {chunk_id} not found")
        except StorageError:
            raise
        except Exception as e:
            logger.error(f"Error deleting chunk {chunk_id}: {e}")
            raise StorageError(f"Failed to delete chunk: {e}")
    
    async def list_documents(self, filter: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        List documents with optional filter.
        
        Args:
            filter: Optional list of status values to filter by.
                   If None, returns all documents.
                   Example: ["uploaded", "error"] to get unprocessed documents.
                   Currently supports status filtering, but can be extended for other filters.
        
        Returns:
            List of document dictionaries with metadata
        """
        try:
            with self.SessionLocal.begin() as session:
                query = session.query(DocumentModel)
                
                # Apply filter if provided (currently supports status filtering)
                if filter:
                    # Convert string statuses to DocumentStatus enums
                    status_enums = [DocumentStatus(status) for status in filter]
                    query = query.filter(DocumentModel.status.in_(status_enums))
                
                docs = query.all()
                result = []
                for doc in docs:
                    # Count chunks for this document
                    chunk_count = len(doc.chunks) if doc.chunks else 0
                    result.append({
                        "doc_id": doc.doc_id,
                        "doc_name": doc.doc_name,
                        "doc_size": doc.doc_size,
                        "upload_date": doc.upload_date.isoformat() if doc.upload_date else None,
                        "status": doc.status.value if isinstance(doc.status, DocumentStatus) else doc.status,
                        "doc_authors": doc.doc_authors,
                        "doc_abstract": doc.doc_abstract,
                        "doc_path": doc.doc_path,
                        "doc_published": doc.doc_published.isoformat() if doc.doc_published else None,
                        "num_chunks": chunk_count
                    })
                return result
        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            raise StorageError(f"Failed to list documents: {e}")
    
    async def get_document_with_chunks(self, doc_id: str) -> Optional[Dict[str, Any]]:
        try:
            with self.SessionLocal.begin() as session:
                doc = session.query(DocumentModel).filter_by(doc_id=doc_id).first()
                if not doc:
                    return None
                
                result = {
                    "doc_id": doc.doc_id,
                    "doc_name": doc.doc_name,
                    "doc_size": doc.doc_size,
                    "upload_date": doc.upload_date.isoformat() if doc.upload_date else None,
                    "status": doc.status.value if isinstance(doc.status, DocumentStatus) else doc.status,
                    "doc_authors": doc.doc_authors,
                    "doc_abstract": doc.doc_abstract,
                    "doc_path": doc.doc_path,
                    "doc_published": doc.doc_published.isoformat() if doc.doc_published else None,
                    "chunks": [
                        {
                            "chunk_id": chunk.chunk_id,
                            "chunk_name": chunk.chunk_name,
                            "chunk_path": chunk.chunk_path,
                            "chunk_source": chunk.chunk_source.value if isinstance(chunk.chunk_source, ChunkSource) else chunk.chunk_source,
                            "chunk_level": chunk.chunk_level.value if isinstance(chunk.chunk_level, ChunkLevel) else chunk.chunk_level
                        }
                        for chunk in doc.chunks
                    ]
                }
                return result
        except Exception as e:
            logger.error(f"Error getting document with chunks {doc_id}: {e}")
            raise StorageError(f"Failed to get document with chunks: {e}")
    
    async def update_document_status(self, doc_id: str, status: str) -> None:
        try:
            # Validate status
            if status not in ["uploaded", "processing", "processed", "error"]:
                raise ValueError(f"Invalid status: {status}. Must be one of: uploaded, processing, processed, error")
            
            with self.SessionLocal.begin() as session:
                doc = session.query(DocumentModel).filter_by(doc_id=doc_id).first()
                if doc:
                    doc.status = DocumentStatus(status)
        except Exception as e:
            logger.error(f"Error updating document status {doc_id}: {e}")
            raise StorageError(f"Failed to update document status: {e}")

