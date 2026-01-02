"""
Agentic query endpoints
"""

from fastapi import APIRouter, HTTPException, Depends
from domain.agentic.orchestrator import AgentOrchestrator
from api.schemas.agent import (
    AgentQueryRequest, 
    AgentQueryResponse, 
    ToolsResponse, 
    ToolInfo,
    RetrievePageChunksRequest,
    RetrievePageChunksResponse,
)
from api.dependencies import get_agent_orchestrator, get_retrieval_service
from services.retrieval_service import RetrievalService

router = APIRouter(prefix="/api/v1/agent", tags=["agent"])


@router.post("/query", response_model=AgentQueryResponse)
async def process_agent_query(
    query_request: AgentQueryRequest,
    orchestrator: AgentOrchestrator = Depends(get_agent_orchestrator)
):
    """
    Process a user query through the agent system.
    
    Args:
        query_request: AgentQueryRequest containing:
            - query: str - The user's query/question
            - messages: Optional[List[LLMMessage]] - Previous conversation history in LLM format.
                       If provided, the new query is appended to continue the conversation.
                       Each message must have a 'role' field ('user', 'assistant', or 'tool')
                       and appropriate content based on role.
    
    Returns:
        AgentQueryResponse containing:
            - messages: List[Dict[str, Any]] - Complete conversation in LLM format, including:
                - User query
                - Assistant tool calls (when LLM decides to use tools)
                - Tool execution results
                - Final assistant answer
    
    The messages array shows the full agentic flow with tool usage visible.
    Messages are in LLM format (what the LLM API expects):
    - {"role": "user", "content": "..."}
    - {"role": "assistant", "content": "..."} or {"role": "assistant", "content": None, "tool_calls": [...]}
    - {"role": "tool", "tool_call_id": "...", "name": "...", "content": "..."}
    
    Raises:
        HTTPException: If processing fails (500 status code with error details)
    """
    try:
        messages_dict = None
        if query_request.messages:
            # Convert Pydantic models to dicts for orchestrator
            messages_dict = [msg.model_dump() for msg in query_request.messages]
        
        messages = await orchestrator.process_query(
            query=query_request.query,
            messages=messages_dict
        )
        return AgentQueryResponse(messages=messages)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tools", response_model=ToolsResponse)
async def list_agent_tools(
    orchestrator: AgentOrchestrator = Depends(get_agent_orchestrator)
):
    """
    List available tools (external MCP tools and internal RAG tools).
    
    Returns:
        ToolsResponse containing:
            - tools: List[ToolInfo] - List of available tools, where each tool has:
                - name: str - Tool name
                - description: str - Tool description for the LLM
                - input_schema: Dict[str, Any] - Tool input schema (JSON schema format)
    
    Raises:
        HTTPException: If tool listing fails (500 status code with error details)
    """
    try:
        all_tools = orchestrator.list_tools()
        tools = [
            ToolInfo(
                name=tool.name,
                description=tool.description,
                input_schema=tool.input_schema,
            )
            for tool in all_tools
        ]
        return ToolsResponse(tools=tools)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/retrieve_chunks", response_model=RetrievePageChunksResponse)
async def retrieve_chunks(
    request: RetrievePageChunksRequest,
    retrieval_service: RetrievalService = Depends(get_retrieval_service)
):
    """
    Retrieve ranked page chunks for one or more queries.
    
    Performs semantic search over the document corpus using the full retrieval pipeline:
    query embedding → ANN (Approximate Nearest Neighbor) search → MaxSim reranking.
    
    Supports both ANN-only retrieval (fast) and reranked retrieval (more accurate).
    Can optionally extract text from PDF chunks for text-based LLMs that cannot process
    PDFs directly.
    
    Args:
        request: RetrievePageChunksRequest containing:
            - queries: List[str] - One or more query text strings to search for
            - use_reranking: bool - Whether to apply MaxSim reranking (default: True).
                                   If False, returns ANN results only.
            - top_k_ann: Optional[int] - Number of candidates from ANN search (default from config).
                                       Used as initial candidate pool before reranking.
            - top_k_rerank: Optional[int] - Number of final results after reranking (default from config).
                                           If use_reranking=False, this is ignored.
            - filter: Optional[Dict[str, Any]] - Metadata filter for ChromaDB (e.g., {"doc_id": "123"}).
                                                 Filters candidates before reranking.
            - force_pdf_to_text: bool - Whether to extract text from PDF chunk files (default: True).
                                       When True, extracts text from PDF files and includes it in
                                       the chunk_text field for text-based LLMs.
        
    Returns:
        RetrievePageChunksResponse containing:
            - results: List[QueryResults] - One result per query (in same order as input queries).
              Each QueryResults contains:
                - query: str - The original query string
                - chunks: List[PageChunkResult] - Ranked list of retrieved chunks, where each chunk has:
                    - chunk_id: str - Unique chunk identifier
                    - chunk_name: str - Page chunk filename (e.g., "document__1.pdf")
                    - score: float - Similarity score (higher is better)
                    - chunk_text: str - Chunk text content. If force_pdf_to_text=True and chunk is PDF,
                                       contains extracted text from the PDF file. Otherwise, contains
                                       text from metadata or empty string.
    
    Raises:
        HTTPException: If retrieval fails (500 status code with error details)
    """
    try:
        page_chunks_by_query = await retrieval_service.retrieve_chunks(
            queries=request.queries,
            top_k_ann=request.top_k_ann,
            top_k_rerank=request.top_k_rerank,
            filter=request.filter,
            use_reranking=request.use_reranking,
            force_pdf_to_text=request.force_pdf_to_text
        )
        
        return RetrievePageChunksResponse(
            results=page_chunks_by_query
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")
