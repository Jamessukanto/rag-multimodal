"""
Evaluation service - orchestrates evaluation
"""

import logging
from typing import List, Dict, Any
from domain.evaluation.evaluator import Evaluator
from domain.evaluation.ground_truth import GroundTruthManager
from domain.evaluation.reporter import EvaluationReporter
from services.retrieval_service import RetrievalService
from services.base import BaseService
from core.exceptions import EvaluationError

logger = logging.getLogger(__name__)


class EvaluationService(BaseService):
    """Orchestrates evaluation"""
    
    def __init__(
        self,
        retrieval_service: RetrievalService
    ):
        self.retrieval_service = retrieval_service
        
        ground_truth_manager = GroundTruthManager()  # No file by default
        self.evaluator = Evaluator(ground_truth_manager)
        self.reporter = EvaluationReporter()
    
    async def run_evaluation(
        self,
        queries: List[str],
        k_values: List[int] = [1, 5, 10]
    ) -> Dict[str, Any]:
        """
        Run evaluation on a set of queries.
        
        Args:
            queries: List of query texts
            k_values: List of k values for metrics
            
        Returns:
            Evaluation results dictionary
        """
        try:
            # Batch embed all queries at once
            query_embedding_results = await self.retrieval_service.embedding_service.generate_query_embeddings(queries)
            
            # Extract all query vectors
            query_vectors = [result.single_vector.embedding for result in query_embedding_results]
            
            # Batch retrieve for all queries at once
            all_candidates = await self.retrieval_service.ann_retriever.retrieve(
                query_vectors=query_vectors,
                top_k=10  # Default top_k for evaluation
            )
            
            # Build retrieved_dict from batch results
            retrieved_dict = {}
            for i, query in enumerate(queries):
                candidates = all_candidates[i] if i < len(all_candidates) else []
                retrieved_dict[query] = [r["chunk_id"] for r in candidates]
            
            # Evaluate
            evaluation_results = self.evaluator.evaluate_batch(
                queries=queries,
                retrieved_dict=retrieved_dict,
                k_values=k_values
            )
            
            return evaluation_results
        except Exception as e:
            logger.error(f"Error in evaluation: {e}")
            raise EvaluationError(f"Evaluation failed: {e}")

