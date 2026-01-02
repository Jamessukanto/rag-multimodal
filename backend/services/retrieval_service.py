"""
Retrieval service - orchestrates retrieval: query embedding → ANN search → reranking
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import fitz  # PyMuPDF
from domain.rag.retrieval.ann_retriever import ANNRetriever
from domain.rag.retrieval.reranker import Reranker
from domain.rag.retrieval.types import RetrievalResult
from storage.single_vector_store import SingleVectorStore
from storage.multi_vector_store import MultiVectorStore
from storage.base import BaseDocumentSQLStore
from services.base import BaseService
from services.embedding_service import EmbeddingService
from core.config import settings
from core.exceptions import RetrievalError

logger = logging.getLogger(__name__)


class RetrievalService(BaseService):
    """
    Orchestrates retrieval pipeline: embed query → ANN search → optionally rerank.
    
    Handles the full retrieval flow from text query to ranked results.
    """
    
    def __init__(
        self,
        single_vector_store: SingleVectorStore,
        multi_vector_store: MultiVectorStore,
        embedding_service: EmbeddingService,
        document_sql_store: Optional[BaseDocumentSQLStore] = None
    ):
        # Instantiate retrieval components
        self.ann_retriever = ANNRetriever(single_vector_store)
        self.reranker = Reranker(multi_vector_store)
        self.embedding_service = embedding_service
        self.document_sql_store = document_sql_store
    
    async def _extract_pdf_text(self, chunk_id: str) -> str:
        """Extract text from a PDF chunk file."""
        try:
            if not self.document_sql_store:
                logger.warning("document_sql_store not available, cannot extract PDF text")
                return ""
            
            chunk_meta = await self.document_sql_store.get_chunk(chunk_id)
            if not chunk_meta or not chunk_meta.get("chunk_path"):
                logger.warning(f"Chunk metadata not found for chunk_id: {chunk_id}")
                return ""
            
            chunk_path = Path(chunk_meta["chunk_path"])
            if not chunk_path.exists():
                logger.warning(f"Chunk PDF file not found: {chunk_path}")
                return ""
            
            # Extract text from PDF using PyMuPDF (fitz)
            with fitz.open(chunk_path) as doc:
                if len(doc) == 0:
                    return ""
                # Get text from first page (page chunks are single-page PDFs)
                page = doc[0]
                return page.get_text()
                
        except Exception as e:
            logger.error(f"Error extracting text from PDF for chunk {chunk_id}: {e}")
            return ""
    
    
    async def retrieve_chunks(
        self,
        queries: List[str],
        top_k_ann: int = None,
        top_k_rerank: int = None,
        filter: Dict[str, Any] = None,
        use_reranking: bool = True,
        force_pdf_to_text: bool = True,
    ) -> List[List[Dict[str, Any]]]:
        """
        Retrieve and rank page chunks for multiple queries.
        
        Performs the full retrieval pipeline: embeds queries, searches vector store,
        optionally reranks results, and extracts text from PDF chunks when needed.
        Results are grouped by query in the same order as input queries.
        
        Args:
            queries: List of query text strings to search for.
            top_k_ann: Number of candidates from ANN search (default from config).
                      Used as initial candidate pool before reranking.
            top_k_rerank: Number of final results after reranking (default from config).
                         If use_reranking=False, this is ignored and top_k_ann results are returned.
            filter: Optional metadata filter for ChromaDB (e.g., {"doc_id": "123"}).
                   Applied before reranking to filter candidates.
            use_reranking: Whether to apply MaxSim reranking using multi-vector embeddings (default: True).
                         If False, returns top_k_ann ANN results only.
            force_pdf_to_text: Whether to extract text from PDF chunk files (default: True).
                             When True and chunk_source is "pdf", extracts text from the PDF file
                             for text-based LLMs that cannot process PDFs directly.
                             The extracted text is included in the 'chunk_text' field of results.
        
        Returns:
            List of dicts, one per query (in same order as input queries).
            Each dict contains:
            - 'query': str - The original query string
            - 'chunks': List[Dict] - Ranked list of chunk results, where each chunk dict contains:
                - 'chunk_id': str - Unique chunk identifier
                - 'chunk_name': str - Page chunk filename (e.g., "document__1.pdf")
                - 'score': float - Similarity score (higher is better)
                - 'chunk_text': str - Chunk text content. If force_pdf_to_text=True and chunk is PDF,
                                     this contains extracted text from the PDF file. Otherwise,
                                     contains text from metadata or empty string.
        """
        try:
            if not queries:
                raise RetrievalError("queries list must not be empty")
            
            top_k_ann = top_k_ann or settings.default_top_k_ann
            top_k_rerank = top_k_rerank or settings.default_top_k_rerank
            
            # Embed all queries at once
            query_embedding_results = await self.embedding_service.generate_query_embeddings(queries)
            query_vectors = [r.single_vector.embedding for r in query_embedding_results]
            
            # Batch ANN search for all queries
            all_candidates = await self.ann_retriever.retrieve(
                query_vectors=query_vectors,
                top_k=top_k_ann,
                filter=filter
            )
            
            # Process each query's results separately and return in order
            results_by_query = []
            
            for query_idx, query_embedding_result in enumerate(query_embedding_results):
                candidates = all_candidates[query_idx] if query_idx < len(all_candidates) else []
                
                if not candidates:
                    results_by_query.append([])
                    continue
                
                # Optionally rerank
                if use_reranking:
                    query_multi_vectors = query_embedding_result.multi_vectors.embeddings
                    if not query_multi_vectors:
                        query_multi_vectors = [query_vectors[query_idx]]
                    
                    reranked = await self.reranker.rerank(
                        query_multi_vectors=query_multi_vectors,
                        candidates=candidates
                    )
                    final_results = reranked[:top_k_rerank]
                else:
                    final_results = candidates[:top_k_ann]
                
                # Format results for this query
                page_chunks = []
                for result in final_results:
                    chunk_id = result.get("chunk_id")
                    metadata = result.get("metadata", {})
                    chunk_name = metadata.get("chunk_name", "")
                    chunk_text = metadata.get("chunk_text", "")
                    chunk_score = result.get("score", 0.0)

                    # Extract text for text-based LLMs that can't process PDFs directly
                    if force_pdf_to_text and metadata.get("chunk_source") == "pdf":
                        chunk_text = await self._extract_pdf_text(chunk_id)
                    
                    page_chunks.append({
                        "chunk_id": chunk_id,
                        "chunk_name": chunk_name,
                        "score": chunk_score,
                        "chunk_text": chunk_text
                    })
                
                results_by_query.append({
                    "query": queries[query_idx],
                    "chunks": page_chunks
                })
            
            return results_by_query
        except Exception as e:
            logger.error(f"Error in retrieve_chunks: {e}")
            raise RetrievalError(f"Page chunk search failed: {e}")

