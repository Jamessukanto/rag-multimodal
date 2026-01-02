"""
Ingestion service - orchestrates download + split + metadata + embedding
INTERNAL SERVICE: Called by API endpoints
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
from domain.rag.ingestion.splitter import PDFSplitter
from storage.base import BaseDocumentSQLStore
from storage.base import BaseFileStore
from storage.document_sql_store import DocumentStatus
from services.base import BaseService
from services.embedding_service import EmbeddingService
from core.exceptions import IngestionError
from core.config import settings

logger = logging.getLogger(__name__)


class IngestionService(BaseService):
    """Orchestrates document ingestion pipeline: split → store chunks → generate embeddings."""
    
    def __init__(
        self,
        document_sql_store: BaseDocumentSQLStore = None,
        file_store: BaseFileStore = None,
        embedding_service: EmbeddingService = None,
    ):
        self.splitter = PDFSplitter(chunk_dir=str(settings.chunks_dir))
        self.document_sql_store = document_sql_store
        self.file_store = file_store
        self.embedding_service = embedding_service
    

    async def ingest_unprocessed_documents(self) -> Dict[str, Any]:
        """Processes all unprocessed documents."""

        if not self.file_store:
            raise IngestionError("FileStore is required for PDF ingestion")
        if not self.document_sql_store:
            raise IngestionError("DocumentSQLStore is required for PDF ingestion")

        result = {
            "num_documents_just_processed": 0,
            "num_chunks_just_processed": 0,
            "num_documents_failed": 0,
            "failed_documents": [],
        }

        try:
            unprocessed_documents = await self.document_sql_store.list_documents(
                filter=[DocumentStatus.UPLOADED.value, DocumentStatus.ERROR.value]
            )
            logger.info(f"Found {len(unprocessed_documents)} unprocessed documents (status: uploaded or error)")
            if not unprocessed_documents:
                logger.info("No unprocessed documents found")
                return result
            
            # For now, loop to process unprocessed documents
            logger.info(f"Processing {len(unprocessed_documents)} documents...")
            for doc in unprocessed_documents:
                doc_id = doc["doc_id"]
                logger.info(f"Processing document {doc_id} ({doc.get('doc_name', 'unknown')})")
                
                try:
                    num_chunks = await self.ingest_document(doc_id)
                    result["num_chunks_just_processed"] += num_chunks
                    result["num_documents_just_processed"] += 1
                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"Error processing document {doc_id}: {error_msg}", exc_info=True)
                    await self._update_document_status(doc_id, DocumentStatus.ERROR.value)
                    result["num_documents_failed"] += 1
                    result["failed_documents"].append({
                        "doc_id": doc_id,
                        "error": error_msg
                    })
                    # Do not re-raise here, allow other documents to be processed

            return result

        except Exception as e:
            logger.error(f"Error during ingestion: {e}", exc_info=True)
            raise IngestionError(f"Failed to ingest documents: {e}")
    

    async def ingest_document(self, doc_id: str) -> int:
        """Process a single document: split → store → embed."""

        if not self.document_sql_store:
            raise IngestionError("DocumentSQLStore is required for document ingestion")
        if not self.file_store:
            raise IngestionError("FileStore is required for document ingestion")
        
        doc_name, pdf_path = await self._get_document_metadata(doc_id)
        await self._update_document_status(doc_id, DocumentStatus.PROCESSING.value)
        
        # Split PDF into chunks, store chunks, store metadata
        chunks = self.splitter.split_pdf_and_store_page_chunks(
            doc_id, str(pdf_path), doc_name
        )
        await self._upsert_chunks_to_document_sql_store(
            doc_id, chunks
        )

        # Generate and store chunk embeddings 
        if not self.embedding_service:
            logger.error(f"EmbeddingService not initialized - cannot generate embeddings for document {doc_id}")
            raise IngestionError("EmbeddingService is required for ingestion. Verify JINA_API_KEY is set in environment variables.")
        
        await self._generate_and_store_chunk_embeddings(doc_id, chunks)

        await self._update_document_status(doc_id, DocumentStatus.PROCESSED.value)
        return len(chunks)
 

    async def _get_document_metadata(self, doc_id: str) -> Tuple[str, Path]:
        """Get document metadata from the database and construct file path."""
        doc = await self.document_sql_store.get_document(doc_id)
        if not doc:
            raise IngestionError(f"Document {doc_id} not found in database")
        
        doc_name = doc["doc_name"]
        pdf_path = self.file_store.get_file_path(doc_id)  # file_store is the source of truth
                
        if not pdf_path.exists():
            await self._update_document_status(doc_id, DocumentStatus.ERROR.value)
            raise IngestionError(f"PDF file not found for {doc_id} at {pdf_path}")
        
        return doc_name, pdf_path
        

    async def _generate_and_store_chunk_embeddings(self, doc_id: str, chunks: List[Dict[str, Any]]) -> None:
        """Generate embeddings for the chunks."""
        try:
            await self.embedding_service.generate_chunk_embeddings(
                chunk_inputs=[{
                    "chunk_id": chunk["chunk_id"],
                    "chunk_source": chunk["chunk_source"],
                } for chunk in chunks],
            )
            logger.info(f"Generated embeddings for {len(chunks)} chunks of document {doc_id}")
        except Exception as e:
            logger.error(f"Failed to generate embeddings for document {doc_id}: {e}", exc_info=True)
            raise IngestionError(f"Embedding generation failed for document {doc_id}: {e}")


    async def _update_document_status(self, doc_id: str, status: str) -> None:
        """Update document status in the database."""
        await self.document_sql_store.update_document_status(doc_id, status)
    

    async def _upsert_chunks_to_document_sql_store(self, doc_id: str, chunks: List[Dict[str, Any]]) -> None:
        """Store chunks in the document SQL store."""
        for chunk in chunks:
            await self.document_sql_store.upsert_chunk(
                chunk_id=chunk["chunk_id"],
                doc_id=doc_id,
                chunk_path=chunk["chunk_path"],
                chunk_name=chunk["chunk_name"],
                chunk_source=chunk["chunk_source"],
                chunk_level=chunk["chunk_level"]
            )
