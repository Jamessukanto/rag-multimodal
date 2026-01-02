"""
Evaluation endpoints
"""

import uuid
from typing import List
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends
from api.schemas.evaluation import (
    RunEvaluationRequest,
    EvaluationResponse,
    EvaluationResult,
    EvaluationHistoryResponse
)
from services.evaluation_service import EvaluationService
from api.dependencies import get_evaluation_service

router = APIRouter(prefix="/api/v1/evaluation", tags=["evaluation"])


@router.post("/run", response_model=EvaluationResponse)
async def run_evaluation(
    request: RunEvaluationRequest,
    service: EvaluationService = Depends(get_evaluation_service)
):
    """Run evaluation"""
    try:
        results = await service.run_evaluation(
            queries=request.queries,
            k_values=request.k_values
        )
        
        run_id = str(uuid.uuid4())
        
        # Format per-query results
        per_query = []
        for r in results.get("per_query", []):
            per_query.append(EvaluationResult(
                query=r["query"],
                num_relevant=r["num_relevant"],
                num_retrieved=r["num_retrieved"],
                mrr=r["mrr"],
                recall_at_k={k: r[f"recall@{k}"] for k in request.k_values},
                ndcg_at_k={k: r[f"ndcg@{k}"] for k in request.k_values}
            ))
        
        return EvaluationResponse(
            run_id=run_id,
            aggregated=results.get("aggregated", {}),
            per_query=per_query
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/results/{run_id}", response_model=EvaluationResponse)
async def get_evaluation_results(run_id: str):
    """Get evaluation results"""
    # TODO: Implement result storage/retrieval
    raise HTTPException(status_code=501, detail="Not implemented")


@router.get("/history", response_model=List[EvaluationHistoryResponse])
async def get_evaluation_history():
    """List evaluation runs"""
    # TODO: Implement history tracking
    raise HTTPException(status_code=501, detail="Not implemented")

