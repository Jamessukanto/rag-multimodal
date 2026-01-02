"""
Application startup and initialization logic
"""

import logging
from fastapi import FastAPI, HTTPException

# Agentic components
from domain.agentic.llm.factory import create_llm_client
from domain.agentic.mcp.client import MCPClient
from domain.agentic.orchestrator import AgentOrchestrator
from domain.agentic.tools.registry import ToolRegistry
from core.config import settings
from core.exceptions import MCPConnectionError, LLMError, EmbeddingError

# RAG components
from storage import SingleVectorStore, MultiVectorStore
from storage.document_sql_store import DocumentSQLStore
from storage.file_store import FileStore
from domain.rag.embedding.client import JinaEmbeddingClient
from services.ingestion_service import IngestionService
from services.embedding_service import EmbeddingService
from services.retrieval_service import RetrievalService
from services.evaluation_service import EvaluationService
from services.document_service import DocumentService

logger = logging.getLogger(__name__)


def raise_startup_error(message: str, error: Exception = None) -> None:
    """Helper to raise HTTPException for startup errors."""
    detail = f"{message}: {error}" if error else message
    raise HTTPException(status_code=500, detail=detail)


async def initialize_agentic_system(app: FastAPI):
    """Initialize LLM, MCPs, Tool Registry, and Agent Orchestrator."""

    # Initialize LLM client
    try:
        llm_client = create_llm_client()
    except LLMError as e:
        raise_startup_error("Failed to initialize LLM client", e)
    
    # Initialize MCP clients
    mcp_client = MCPClient()
    try:
        await mcp_client.connect_to_server(settings.mcp_server_script_path)
    except MCPConnectionError as e:
        raise_startup_error("Failed to connect to MCP server", e)
    
    # Initialize tool registry
    tool_registry = ToolRegistry()
    tool_registry.register_external_tools(mcp_client)  
    
    # Initialize orchestrator
    agent_orchestrator = AgentOrchestrator(
        llm_client=llm_client,
        tool_registry=tool_registry,
    )
    
    app.state.mcp_client = mcp_client  
    app.state.tool_registry = tool_registry 
    app.state.agent_orchestrator = agent_orchestrator  


async def initialize_rag_system(app: FastAPI):
    """
    Initialize RAG components (storage, retrieval, services).
    Also registers RAG tools in the tool registry.
    """
    # Initialize stores
    single_vector_store = SingleVectorStore()
    multi_vector_store = MultiVectorStore()
    document_sql_store = DocumentSQLStore()
    file_store = FileStore()
    
    # Initialize RAG services
    document_service = DocumentService(
        file_store=file_store,
        document_sql_store=document_sql_store,
        single_vector_store=single_vector_store,
        multi_vector_store=multi_vector_store
    )
    embedding_service = EmbeddingService(
        document_sql_store=document_sql_store,
        single_vector_store=single_vector_store,
        multi_vector_store=multi_vector_store,
        chunk_embedding_client=JinaEmbeddingClient(task="retrieval.passage"),
        query_embedding_client=JinaEmbeddingClient(task="retrieval.query"),
    )
    ingestion_service = IngestionService(
        document_sql_store=document_sql_store,
        file_store=file_store,
        embedding_service=embedding_service,
    )
    retrieval_service = RetrievalService(
        single_vector_store=single_vector_store,
        multi_vector_store=multi_vector_store,
        embedding_service=embedding_service,
        document_sql_store=document_sql_store,
    )
    evaluation_service = EvaluationService(
        retrieval_service
    )
    
    # Register internal tools for agentic system
    if hasattr(app.state, 'tool_registry'):
        app.state.tool_registry.register_internal_tools(retrieval_service)
    
    app.state.single_vector_store = single_vector_store
    app.state.multi_vector_store = multi_vector_store
    app.state.document_sql_store = document_sql_store
    app.state.file_store = file_store

    app.state.embedding_service = embedding_service 
    app.state.document_service = document_service
    app.state.ingestion_service = ingestion_service
    app.state.retrieval_service = retrieval_service
    app.state.evaluation_service = evaluation_service


async def cleanup_agentic_system(app: FastAPI):
    """Cleanup agentic system resources (MCP client)."""
    if hasattr(app.state, 'mcp_client') and app.state.mcp_client:
        try:
            await app.state.mcp_client.cleanup()
            logger.info("MCP client cleaned up")
        except Exception as e:
            logger.error(f"Error during MCP client cleanup: {e}", exc_info=True)


async def cleanup_rag_system(app: FastAPI):
    """Cleanup RAG system resources (embedding service HTTP connections)."""
    if hasattr(app.state, 'embedding_service') and app.state.embedding_service:
        try:
            await app.state.embedding_service.close()
            logger.info("Embedding service cleaned up")
        except Exception as e:
            logger.error(f"Error during embedding service cleanup: {e}", exc_info=True)
