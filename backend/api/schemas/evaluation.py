"""
Pydantic models for evaluation endpoints
"""

from pydantic import BaseModel
from typing import List, Dict, Any, Optional


class RunEvaluationRequest(BaseModel):
    """Request to run evaluation"""
    queries: List[str]
    k_values: List[int] = [1, 5, 10]


class EvaluationResult(BaseModel):
    """Evaluation result for a single query"""
    query: str
    num_relevant: int
    num_retrieved: int
    mrr: float
    recall_at_k: Dict[int, float]
    ndcg_at_k: Dict[int, float]


class EvaluationResponse(BaseModel):
    """Response from evaluation"""
    run_id: str
    aggregated: Dict[str, Any]
    per_query: List[EvaluationResult]


class EvaluationHistoryResponse(BaseModel):
    """Evaluation run history"""
    run_id: str
    timestamp: str
    num_queries: int
    aggregated_metrics: Dict[str, float]

