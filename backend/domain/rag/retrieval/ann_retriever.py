"""
ANN retrieval using single vectors
"""

import logging
from typing import List, Dict, Any
from domain.rag.retrieval.types import RetrievalResult
from storage.single_vector_store import SingleVectorStore
from core.exceptions import RetrievalError

logger = logging.getLogger(__name__)


class ANNRetriever:
    """ANN search using single vectors"""
    
    def __init__(self, vector_store: SingleVectorStore):
        self.vector_store = vector_store
    
    async def retrieve(
        self,
        query_vectors: List[List[float]],
        top_k: int = 10,
        filter: Dict[str, Any] = None
    ) -> List[List[Dict[str, Any]]]:
        """
        Retrieve documents using ANN search for multiple query vectors.
        
        Args:
            query_vectors: List of query embedding vectors
            top_k: Number of results to return per query
            filter: Optional metadata filter
            
        Returns:
            List of result lists, one per query vector. Each inner list contains dicts matching RetrievalResult structure:
            - 'chunk_id': str - Unique chunk identifier
            - 'score': float - Similarity score (higher is better)
            - 'metadata': Dict[str, Any] - Chunk metadata (includes 'chunk_name' and other fields)
        """
        try:
            if not query_vectors:
                raise RetrievalError("query_vectors list must not be empty")
            
            # Query vector store with all query vectors at once
            results = await self.vector_store.query(
                query_vectors=query_vectors,
                top_k=top_k,
                filter=filter
            )
            return results
        except Exception as e:
            logger.error(f"Error in ANN retrieval: {e}")
            raise RetrievalError(f"ANN retrieval failed: {e}")

