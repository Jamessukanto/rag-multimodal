"""
FastAPI dependencies
"""

from fastapi import Request, HTTPException
from domain.agentic.orchestrator import AgentOrchestrator
from services.document_service import DocumentService
from services.ingestion_service import IngestionService
from services.embedding_service import EmbeddingService
from services.retrieval_service import RetrievalService
from services.evaluation_service import EvaluationService


def get_agent_orchestrator(request: Request) -> AgentOrchestrator:
    return request.app.state.agent_orchestrator


def get_document_service(request: Request) -> DocumentService:
    return request.app.state.document_service


def get_ingestion_service(request: Request) -> IngestionService:
    return request.app.state.ingestion_service


def get_embedding_service(request: Request) -> EmbeddingService:
    return request.app.state.embedding_service


def get_retrieval_service(request: Request) -> RetrievalService:
    return request.app.state.retrieval_service


def get_evaluation_service(request: Request) -> EvaluationService:
    return request.app.state.evaluation_service

