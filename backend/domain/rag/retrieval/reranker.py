"""
MaxSim reranking using multi-vectors
"""

import logging
from typing import List, Dict, Any
from domain.rag.retrieval.types import RetrievalResult
from storage.multi_vector_store import MultiVectorStore
from domain.rag.retrieval.similarity import maxsim_score
from core.exceptions import RetrievalError

logger = logging.getLogger(__name__)


class Reranker:
    """MaxSim scoring using multi-vectors"""
    
    def __init__(self, multi_vector_store: MultiVectorStore):
        self.multi_vector_store = multi_vector_store
    
    async def rerank(
        self,
        query_multi_vectors: List[List[float]],
        candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Rerank candidates using MaxSim scoring.
        
        Args:
            query_multi_vectors: List of query token embedding vectors
            candidates: List of candidate results from ANN search
            
        Returns:
            List of dicts matching RetrievalResult structure:
            - 'chunk_id': str - Unique chunk identifier
            - 'score': float - MaxSim score (higher is better)
            - 'metadata': Dict[str, Any] - Chunk metadata (includes 'chunk_name' and other fields, preserved from input)
        """
        try:
            # Get multi-vectors for all candidates
            chunk_ids = [c["chunk_id"] for c in candidates]
            chunk_multi_vectors_dict = await self.multi_vector_store.batch_get(chunk_ids)
            
            # Compute MaxSim scores
            reranked = []
            for candidate in candidates:
                chunk_id = candidate["chunk_id"]
                chunk_multi_vectors = chunk_multi_vectors_dict.get(chunk_id)
                
                if chunk_multi_vectors:
                    maxsim = maxsim_score(query_multi_vectors, chunk_multi_vectors)
                    # Create new dict to avoid mutating original, preserving all fields
                    reranked_candidate = {
                        "chunk_id": chunk_id,
                        "score": maxsim,
                        "metadata": candidate.get("metadata", {})
                    }
                    reranked.append(reranked_candidate)
                else:
                    # Fallback to original candidate if no multi-vectors (preserve all fields)
                    reranked.append(candidate.copy())
            
            # Sort by MaxSim score (descending)
            reranked.sort(key=lambda x: x["score"], reverse=True)
            
            return reranked
        except Exception as e:
            logger.error(f"Error in reranking: {e}")
            raise RetrievalError(f"Reranking failed: {e}")

