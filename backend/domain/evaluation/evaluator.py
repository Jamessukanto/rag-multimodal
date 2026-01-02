"""
Evaluation orchestrator
"""

import logging
from typing import List, Dict, Any
from domain.evaluation.metrics import recall_at_k, mrr, ndcg_at_k
from domain.evaluation.ground_truth import GroundTruthManager
from core.exceptions import EvaluationError

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluation orchestrator"""
    
    def __init__(self, ground_truth_manager: GroundTruthManager):
        self.ground_truth = ground_truth_manager
    
    def evaluate_query(
        self,
        query: str,
        retrieved: List[str],
        k_values: List[int] = [1, 5, 10]
    ) -> Dict[str, Any]:
        """
        Evaluate retrieval results for a single query.
        
        Args:
            query: Query text
            retrieved: List of retrieved document/chunk IDs (ordered)
            k_values: List of k values for Recall@k and nDCG@k
            
        Returns:
            Dictionary with metrics
        """
        if not self.ground_truth.has_ground_truth(query):
            raise EvaluationError(f"No ground truth for query: {query}")
        
        relevant = self.ground_truth.get_relevant(query)
        
        metrics = {
            "query": query,
            "num_relevant": len(relevant),
            "num_retrieved": len(retrieved),
            "mrr": mrr(relevant, retrieved),
        }
        
        # Compute Recall@k and nDCG@k for each k
        for k in k_values:
            metrics[f"recall@{k}"] = recall_at_k(relevant, retrieved, k)
            metrics[f"ndcg@{k}"] = ndcg_at_k(relevant, retrieved, k)
        
        return metrics
    
    def evaluate_batch(
        self,
        queries: List[str],
        retrieved_dict: Dict[str, List[str]],
        k_values: List[int] = [1, 5, 10]
    ) -> Dict[str, Any]:
        """
        Evaluate retrieval results for multiple queries.
        
        Args:
            queries: List of query texts
            retrieved_dict: Dictionary mapping query -> list of retrieved IDs
            k_values: List of k values for metrics
            
        Returns:
            Dictionary with aggregated metrics
        """
        results = []
        for query in queries:
            if query in retrieved_dict:
                try:
                    metrics = self.evaluate_query(
                        query,
                        retrieved_dict[query],
                        k_values
                    )
                    results.append(metrics)
                except EvaluationError as e:
                    logger.warning(f"Skipping {query}: {e}")
                    continue
        
        if not results:
            raise EvaluationError("No valid evaluation results")
        
        # Aggregate metrics
        aggregated = {
            "num_queries": len(results),
            "mrr": sum(r["mrr"] for r in results) / len(results),
        }
        
        for k in k_values:
            aggregated[f"recall@{k}"] = sum(r[f"recall@{k}"] for r in results) / len(results)
            aggregated[f"ndcg@{k}"] = sum(r[f"ndcg@{k}"] for r in results) / len(results)
        
        return {
            "aggregated": aggregated,
            "per_query": results
        }

