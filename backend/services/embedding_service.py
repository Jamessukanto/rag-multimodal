"""
Embedding service - orchestrates embedding generation
INTERNAL SERVICE: Called by IngestionService (chunk embeddings) and RetrievalService (query embeddings)
"""

import logging
import base64
from typing import List, Dict, Any, Optional
from domain.rag.embedding.client import JinaEmbeddingClient
from domain.rag.embedding.batch_processor import BatchProcessor
from domain.rag.embedding.types import EmbeddingResult
from storage.base import BaseDocumentSQLStore, BaseSingleVectorStore, BaseMultiVectorStore
from services.base import BaseService
from core.exceptions import EmbeddingError

logger = logging.getLogger(__name__)


class EmbeddingService(BaseService):
    """
    Orchestrates embedding generation for both chunks and queries.
    
    Handles:
    - Chunk embeddings (for document ingestion): builds records, generates embeddings, stores in vector DBs
    - Query embeddings (for retrieval): generates query embeddings for search
    """
    
    # Fields to exclude from vector store metadata
    # These are either: (1) passed separately (chunk_id), (2) too large (chunk_pdf, chunk_text),
    # (3) not useful for search/filtering (file paths), or (4) can become stale (status)
    METADATA_EXCLUDED_FIELDS = {
        'chunk_id',      # Passed as separate parameter to vector store
        'chunk_pdf',     # Base64-encoded PDF data, too large for metadata
        'chunk_text',    # Full text content, too large for metadata
        'doc_path',      # File system path, not useful for search/filtering
        'chunk_path',    # File system path, not useful for search/filtering
        'status',        # Document status can become stale (fetched before status update to "processed")
    }
    
    def __init__(
        self,
        document_sql_store: Optional[BaseDocumentSQLStore] = None,
        single_vector_store: Optional[BaseSingleVectorStore] = None,
        multi_vector_store: Optional[BaseMultiVectorStore] = None,
        chunk_embedding_client: Optional[JinaEmbeddingClient] = None,
        query_embedding_client: Optional[JinaEmbeddingClient] = None,
    ):
        self.chunk_embedding_client = chunk_embedding_client or JinaEmbeddingClient(task="retrieval.passage")
        self.query_embedding_client = query_embedding_client or JinaEmbeddingClient(task="retrieval.query")
        self.document_sql_store = document_sql_store
        self.single_vector_store = single_vector_store
        self.multi_vector_store = multi_vector_store
    

    async def generate_chunk_embeddings(
        self, chunk_inputs: List[Dict[str, Any]]
    ) -> List[EmbeddingResult]:
        """
        Generate embeddings for document chunks (for ingestion).
        
        Args:
            chunk_inputs: List of chunk input dicts with 'chunk_id' and 'chunk_source'
            
        Returns:
            List of EmbeddingResult, one per chunk input
        """

        if not self.document_sql_store or not self.single_vector_store or not self.multi_vector_store:
            raise EmbeddingError("DocumentSQLStore, SingleVectorStore, and MultiVectorStore must be provided")

        if not chunk_inputs or len(chunk_inputs) == 0:
            raise EmbeddingError("chunk_inputs must be provided and not empty")

        try:
            logger.info(f"Generating embeddings for {len(chunk_inputs)} chunks")
            embedables = []
            chunk_records = await self._build_chunk_records(chunk_inputs)

            for chunk_record in chunk_records:
                embedable = {'id': chunk_record["chunk_id"]}
                if "chunk_text" in chunk_record:
                    embedable["text"] = chunk_record["chunk_text"]
                if "chunk_pdf" in chunk_record:
                    embedable["pdf"] = chunk_record["chunk_pdf"]
                embedables.append(embedable)

            logger.info(f"Calling Jina embedding API for {len(embedables)} embedables")
            embedding_results = await self.chunk_embedding_client.embed(embedables)
            await self._store_embeddings(chunk_records, embedding_results)
            logger.info("Successfully stored embeddings in vector stores")
            return embedding_results

        except Exception as e:
            logger.error(f"Error in generate_chunk_embeddings: {e}")
            raise EmbeddingError(f"Failed to generate chunk embeddings: {e}")

    async def generate_query_embeddings(
        self, queries: List[str]
    ) -> List[EmbeddingResult]:
        """Generate embeddings for queries (for retrieval)."""
        try:
            if not queries:
                raise EmbeddingError("queries list must not be empty")
            
            # Build embedables from queries
            embedables = [{"id": f"query_{i}", "text": query} for i, query in enumerate(queries)]
            embedding_results = await self.query_embedding_client.embed(embedables)
            
            if not embedding_results or len(embedding_results) != len(queries):
                raise EmbeddingError(f"Expected {len(queries)} embedding results, got {len(embedding_results)}")
            
            return embedding_results

        except Exception as e:
            logger.error(f"Error in generate_query_embeddings: {e}")
            raise EmbeddingError(f"Failed to generate_query_embeddings: {e}")
    

    async def _build_chunk_records(self, chunk_inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract chunk records from chunk inputs"""

        chunk_records = []
        for chunk_input in chunk_inputs:      
            id = chunk_input.get("chunk_id")
            chunk_record = chunk_input.copy()

            if chunk_input.get("chunk_source") == "pdf":
                # Get metadata from document metadata store
                doc_meta = await self.document_sql_store.get_document_by_chunk(id)
                chunk_meta = await self.document_sql_store.get_chunk(id)

                if not chunk_meta:
                    raise EmbeddingError(f"Chunk {id} not found in database")
                if not doc_meta:
                    raise EmbeddingError(f"Document for chunk {id} not found in database")

                with open(chunk_meta["chunk_path"], "rb") as f:
                    pdf_bytes = f.read()
                pdf_base64 = base64.b64encode(pdf_bytes).decode("utf-8")

                chunk_record = {
                    **doc_meta,
                    **chunk_meta,
                    "chunk_pdf": pdf_base64,
                }

            chunk_records.append(chunk_record)
        return chunk_records


    async def _store_embeddings(self, chunk_records: List[Dict[str, Any]], embedding_results: List[EmbeddingResult]):
        """Store embeddings in the stores"""
        logger.info(f"Storing embeddings for {len(chunk_records)} chunks in vector stores")

        # Store single vectors
        for i, result in enumerate(embedding_results):
            chunk_id = chunk_records[i].get("chunk_id")
            # Build metadata: exclude None values (ChromaDB requirement) and excluded fields
            metadata = {
                k: v for k, v in chunk_records[i].items() 
                if k not in self.METADATA_EXCLUDED_FIELDS and v is not None
            }
            
            await self.single_vector_store.add(
                chunk_id=chunk_id,
                embedding=result.single_vector.embedding,
                metadata=metadata
            )
        logger.info(f"Stored {len(embedding_results)} single vectors in ChromaDB")
        
        # Store multi-vectors
        for i, result in enumerate(embedding_results):
            chunk_id = chunk_records[i].get("chunk_id")
            await self.multi_vector_store.add(
                chunk_id=chunk_id,
                embeddings=result.multi_vectors.embeddings
            )
        logger.info(f"Stored {len(embedding_results)} multi-vectors in file store")


    async def close(self):
        """Close embedding clients"""
        if self.chunk_embedding_client:
            await self.chunk_embedding_client.close()
        if self.query_embedding_client:
            await self.query_embedding_client.close()

