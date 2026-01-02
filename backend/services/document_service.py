"""
Document service - orchestrates document lifecycle management
"""

import logging
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from io import BytesIO

import fitz  # PyMuPDF

from storage.base import BaseFileStore
from storage.base import BaseDocumentSQLStore, BaseSingleVectorStore, BaseMultiVectorStore
from services.base import BaseService
from core.config import settings
from core.exceptions import StorageError

logger = logging.getLogger(__name__)


class DocumentService(BaseService):
    """
    Orchestrates document management: upload, delete, list, get.
    
    Integrates:
    - FileStore: PDF file storage
    - DocumentSQLStore: Document metadata storage
    - Vector stores: For embedding deletion
    """
    
    def __init__(
        self,
        file_store: BaseFileStore,
        document_sql_store: BaseDocumentSQLStore,
        single_vector_store: Optional[BaseSingleVectorStore] = None,
        multi_vector_store: Optional[BaseMultiVectorStore] = None
    ):
        self.file_store = file_store
        self.document_sql_store = document_sql_store
        self.single_vector_store = single_vector_store
        self.multi_vector_store = multi_vector_store
    

    def _extract_pdf_metadata(self, pdf_bytes: bytes) -> Dict[str, Any]:
        try:
            doc = fitz.open(stream=BytesIO(pdf_bytes), filetype="pdf")
            metadata = doc.metadata
            authors = metadata.get("authors", "").strip() or metadata.get("author", "").strip()
            abstract = metadata.get("summary", "").strip() or metadata.get("abstract", "").strip()
            
            # PDF metadata dates are strings, not datetime objects
            # We don't reliably get publication dates from PDF metadata
            published = None
            
            doc.close()
            
            return {
                "authors": authors or "",
                "abstract": abstract or "",
                "published": published
            }
        except Exception as e:
            logger.warning(f"Failed to extract PDF metadata: {e}")
            return {
                "authors": "",
                "abstract": "",
                "published": None
            }


    async def upload_document(
        self,
        file_content_bytes: bytes,
        doc_name: str
    ) -> Dict[str, Any]:
        """
        Upload a PDF document.
        
        Args:
            file_content_bytes: The content of the PDF file.
            doc_name: The name of the document.

        Returns:
            Dictionary with document ID, name, size, upload date, status, path, authors, abstract, and published date

        Raises:
            StorageError: If the file content is empty or if the file size exceeds the maximum size
            Exception: If an unexpected error occurs
        """
        try:
            if not file_content_bytes:
                raise StorageError("File content is empty")
            
            # Check file size (max 50MB by default)
            max_size = settings.max_pdf_size_mb * 1024 * 1024
            doc_size = len(file_content_bytes)
            if doc_size > max_size:
                raise StorageError(f"File size exceeds maximum size ({settings.max_pdf_size_mb}MB)")
            
            # Basic PDF header validation
            if not file_content_bytes.startswith(b'%PDF'):
                raise StorageError("Invalid PDF file")
            
            # Store PDF file
            doc_id = str(uuid.uuid4())
            file_path = await self.file_store.save_file(doc_id, file_content_bytes)
            pdf_metadata = self._extract_pdf_metadata(file_content_bytes)

            result = {
                "doc_id": doc_id,
                "doc_name": doc_name,
                "doc_size": doc_size,
                "upload_date": datetime.utcnow().isoformat(),
                "status": "uploaded",
                "doc_path": str(file_path),
                "doc_authors": pdf_metadata["authors"],
                "doc_abstract": pdf_metadata["abstract"],
                "doc_published": pdf_metadata["published"]
            }
            
            # Store document record
            await self.document_sql_store.upsert_document(**result)
            logger.info(f"Uploaded document {doc_id}: {doc_name} ({doc_size} bytes)")
            
            return result

        except Exception as e:
            logger.error(f"Error uploading document: {e}", exc_info=True)
            raise StorageError(f"Failed to upload document: {str(e)}")
    

    async def delete_document(self, doc_id: str) -> Dict[str, Any]:
        """
        Delete document and all associated data.
        
        Critical operations (failures abort the operation):
        - Database deletion (document + chunks metadata via cascade)
        
        Best-effort operations (failures are logged but don't abort):
        - PDF file deletion
        - Chunk file deletions
        - Embedding deletions from vector stores

        Args:
            doc_id: The ID of the document to delete.
        
        Returns:
            Dictionary with document ID and status

        Raises:
            StorageError: If the document is not found or if the deletion fails
            Exception: If an unexpected error occurs
        """
        try:
            doc_info = await self.document_sql_store.get_document_with_chunks(doc_id)
            if not doc_info:
                raise StorageError(f"Document {doc_id} not found")
            
            chunk_path_files_to_delete = [
                path for chunk in doc_info.get("chunks", [])
                if chunk.get("chunk_path") and (path := Path(chunk["chunk_path"])).exists()
            ]
            chunk_ids = [
                chunk.get("chunk_id") for chunk in doc_info.get("chunks", []) 
                if chunk.get("chunk_id")
            ]
            
            # Delete document metadata (cascades to chunks metadata in DB)
            # If this fails, the entire operation should fail
            await self.document_sql_store.delete_document(doc_id)
            
            # Delete document file
            # TODO: Use a background cleanup job in production
            try:
                await self.file_store.delete_file(doc_id)
            except Exception as e:
                logger.warning(
                    f"Deleting document file '{doc_id}' failed (database already deleted): {e}. "
                    "Clean up orphaned file later."
                )
            
            # Delete chunk files
            # TODO: use a background cleanup job in production
            for chunk_path in chunk_path_files_to_delete:
                try:
                    chunk_path.unlink()
                    logger.debug(f"Deleted chunk file: {chunk_path}")
                except Exception as e:
                    logger.warning(
                        f"Deleting chunk file '{chunk_path}' failed: {e}. "
                        "Clean up orphaned file later."
                    )
            
            # Delete embeddings from vector stores
            # TODO: use a background cleanup job in production
            if chunk_ids:
                # Delete from single vector store (ChromaDB)
                if self.single_vector_store:
                    for chunk_id in chunk_ids:
                        try:
                            await self.single_vector_store.delete(chunk_id)
                            logger.debug(f"Deleted embedding from SingleVectorStore for chunk {chunk_id}")
                        except Exception as e:
                            logger.warning(
                                f"Deleting embedding from SingleVectorStore for chunk '{chunk_id}' failed: {e}. "
                                "Clean up orphaned embedding later."
                            )
                
                # Delete from multi vector store (mmap files)
                if self.multi_vector_store:
                    for chunk_id in chunk_ids:
                        try:
                            await self.multi_vector_store.delete(chunk_id)
                            logger.debug(f"Deleted embedding from MultiVectorStore for chunk {chunk_id}")
                        except Exception as e:
                            logger.warning(
                                f"Deleting embedding from MultiVectorStore for chunk '{chunk_id}' failed: {e}. "
                                "Clean up orphaned embedding later."
                            )
            
            logger.info(f"Deleted document {doc_id} and {len(chunk_path_files_to_delete)} chunk files")
            
            return {
                "doc_id": doc_id,
                "status": "deleted"
            }
        except StorageError:
            # Re-raise StorageErrors as-is (e.g., "document not found")
            raise
        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {e}", exc_info=True)
            raise StorageError(f"Failed to delete document {doc_id}: {e}")
    

    async def delete_chunk(self, chunk_id: str) -> Dict[str, Any]:
        """
        Delete chunk and all associated data.
        
        Critical operations (failures abort the operation):
        - Database deletion (chunk metadata)
        
        Best-effort operations (failures are logged but don't abort):
        - Chunk file deletion
        - Embedding deletions from vector stores

        Args:
            chunk_id: The ID of the chunk to delete.
        
        Returns:
            Dictionary with chunk ID and status
        """
        try:
            chunk_info = await self.document_sql_store.get_chunk(chunk_id)
            if not chunk_info:
                raise StorageError(f"Chunk {chunk_id} not found")
            
            chunk_path = chunk_info.get("chunk_path")
            
            # Delete chunk metadata from database first
            # If this fails, the entire operation should fail
            await self.document_sql_store.delete_chunk(chunk_id)
            
            # Delete chunk file
            # TODO: use a background cleanup job in production
            if chunk_path:
                chunk_file_path = Path(chunk_path)
                if chunk_file_path.exists():
                    try:
                        chunk_file_path.unlink()
                        logger.debug(f"Deleted chunk file: {chunk_file_path}")
                    except Exception as e:
                        logger.warning(
                            f"Deleting chunk file '{chunk_file_path}' failed (database already deleted): {e}. "
                            "Clean up orphaned file later."
                        )

            # Delete from single vector store (ChromaDB)
            # TODO: use a background cleanup job in production
            if self.single_vector_store:
                try:
                    await self.single_vector_store.delete(chunk_id)
                    logger.debug(f"Deleted embedding from SingleVectorStore for chunk {chunk_id}")
                except Exception as e:
                    logger.warning(
                        f"Deleting embedding from SingleVectorStore for chunk '{chunk_id}' failed: {e}. "
                        "Clean up orphaned embedding later."
                    )
            
            # Delete from multi vector store (mmap files)
            # TODO: use a background cleanup job in production
            if self.multi_vector_store:
                try:
                    await self.multi_vector_store.delete(chunk_id)
                    logger.debug(f"Deleted embedding from MultiVectorStore for chunk {chunk_id}")
                except Exception as e:
                    logger.warning(
                        f"Deleting embedding from MultiVectorStore for chunk '{chunk_id}' failed: {e}. "
                        "Clean up orphaned embedding later."
                    )
            
            logger.info(f"Deleted chunk {chunk_id}")
            
            return {
                "chunk_id": chunk_id,
                "status": "deleted"
            }
        except StorageError:
            # Re-raise StorageErrors as-is (e.g., "chunk not found")
            raise
        except Exception as e:
            logger.error(f"Error deleting chunk {chunk_id}: {e}", exc_info=True)
            raise StorageError(f"Failed to delete chunk {chunk_id}: {e}")
    

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
            documents = await self.document_sql_store.list_documents(filter=filter)
            return documents
        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            raise StorageError(f"Failed to list documents: {e}")
    

    async def get_document(self, doc_id: str) -> Dict[str, Any]:
        """
        Get document with all its chunks.
        
        Args:
            doc_id: The ID of the document to get.
        
        Returns:
            Document dictionary with metadata and chunks
        """
        try:
            doc_info = await self.document_sql_store.get_document_with_chunks(doc_id)
            if not doc_info:
                raise StorageError(f"Document {doc_id} not found")
            return doc_info
        except Exception as e:
            logger.error(f"Error getting document {doc_id}: {e}")
            raise StorageError(f"Failed to get document: {e}")

