"""
RAG retrieval tool (internal) - direct function calls
"""

import logging
from typing import Dict, Any, Optional
from domain.agentic.tools.base import BaseTool
from services.retrieval_service import RetrievalService
from core.exceptions import ToolExecutionError

logger = logging.getLogger(__name__)

TOP_K_ANN = 10
TOP_K_RERANK = 5
USE_RERANKING = True
FORCE_PDF_TO_TEXT = True

TEXT_PREVIEW_LENGTH = 500


class RetrieveDocumentsTool(BaseTool):
    """
    RAG retrieval tool for semantic document search.
    
    Performs embedding-based semantic search over ingested documents using a two-stage
    retrieval pipeline: initial ANN (Approximate Nearest Neighbor) search followed by
    optional MaxSim reranking for higher precision. Extracts text from PDF chunks when
    needed for text-based LLMs.
    """
    
    def __init__(self, retrieval_service: RetrievalService):
        self.retrieval_service = retrieval_service
    
    @property
    def name(self) -> str:
        return "retrieve_documents"
    
    @property
    def description(self) -> str:
        return (
            "Retrieve relevant document chunks from the ingested corpus using semantic search. "
            "This tool performs embedding-based retrieval over your document collection using a "
            "two-stage pipeline: fast ANN (Approximate Nearest Neighbor) search to find candidates, "
            "then MaxSim reranking for higher precision. "
            "\n\n"
            "Returns formatted results with:\n"
            "- Document names (e.g., 'document__1.pdf')\n"
            "- Content previews (first 500 characters of chunk text)\n"
            "- Relevance scores (higher is better, typically 0.0 to 5.0+)\n"
            "\n"
            "Use this tool when you need to find information from documents "
            "that have been ingested into the system. The tool automatically extracts text from "
            "PDF chunks for text-based processing."
        )
    
    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to find semantically similar documents. "
                                 "This is the only parameter used."
                }
            },
            "required": ["query"]
        }
    
    async def execute(
        self,
        query: str,
        **kwargs
    ) -> str:
        """
        Execute semantic document retrieval.
        
        Args:
            query: Search query string.
            **kwargs: Unused (required by BaseTool interface).
        
        Returns:
            Formatted text string with retrieval results (document names, content previews, scores).
            Returns "No results found for query: <query>" if no matches.
        
        Raises:
            ToolExecutionError: If retrieval fails.
        """
        try:
            # results_by_query is of type List[Dict]. Each dict has:
            # - "query": str - the original query
            # - "chunks": List[Dict] - Each chunk has chunk_id, chunk_name, score, chunk_text
            results_by_query = await self.retrieval_service.retrieve_chunks(
                queries=[query],  # List[str] - single query in a list
                top_k_ann=TOP_K_ANN,
                top_k_rerank=TOP_K_RERANK,
                use_reranking=USE_RERANKING,
                force_pdf_to_text=FORCE_PDF_TO_TEXT,
            )

            if not results_by_query or len(results_by_query) == 0:
                return f"No results found for query: {query}"
            
            # Get first query result
            query_result = results_by_query[0].get("chunks", [])
            if not query_result:
                return f"No results found for query: {query}"

            # Format results as text for LLM
            search_type = "reranked" if USE_RERANKING else "ANN-only"
            formatted_result_parts = [
                f"Found {len(query_result)} results ({search_type}) for query: {query}\n"
            ]
            
            # Format results as text for LLM
            for i, chunk in enumerate(query_result, 1):
                chunk_name = chunk.get("chunk_name", "")
                chunk_score = chunk.get("score", 0.0)
                chunk_text = chunk.get("chunk_text", "")
                
                result_part = f"\n\nResult {i} (relevance score: {chunk_score:.4f}):"

                if chunk_name:
                    result_part += f"\n  Document: {chunk_name}"
                if chunk_text:
                    text_preview = chunk_text[:TEXT_PREVIEW_LENGTH] \
                        + "..." if len(chunk_text) > TEXT_PREVIEW_LENGTH else chunk_text
                    result_part += f"\n  Content: {text_preview}"
                    
                formatted_result_parts.append(result_part)
            
            return "\n".join(formatted_result_parts)

        except Exception as e:
            logger.error(f"Error in document retrieval: {e}")
            raise ToolExecutionError(f"Document retrieval failed: {e}")

