"""
Pydantic models for agent-related request/response schemas
Refactored from models.py
"""

from pydantic import BaseModel, field_validator, Field, ConfigDict
from typing import Dict, Any, List, Optional, Union, Literal


class LLMToolCallFunction(BaseModel):
    """Function details in a tool call (Groq/OpenAI format)"""
    name: str
    arguments: str  # JSON string


class LLMToolCall(BaseModel):
    """Tool call in LLM format"""
    id: str
    type: Literal["function"] = "function"
    function: LLMToolCallFunction


class LLMUserMessage(BaseModel):
    """User message in LLM format"""
    role: Literal["user"] = Field(..., description="Message role must be 'user'")
    content: str = Field(..., description="User message content")


class LLMAssistantMessage(BaseModel):
    """Assistant message in LLM format (text or tool calls)"""
    role: Literal["assistant"] = Field(..., description="Message role must be 'assistant'")
    content: Optional[str] = Field(None, description="Assistant text content (None if tool_calls present)")
    tool_calls: Optional[List[LLMToolCall]] = Field(None, description="List of tool calls (None if content present)")


class LLMToolMessage(BaseModel):
    """Tool result message in LLM format"""
    role: Literal["tool"] = Field(..., description="Message role must be 'tool'")
    tool_call_id: str = Field(..., description="ID of the tool call this result corresponds to")
    name: str = Field(..., description="Name of the tool that was called")
    content: str = Field(..., description="Tool execution result content")


# Union type for all LLM message formats
# Pydantic will automatically discriminate based on the 'role' Literal field
LLMMessage = Union[LLMUserMessage, LLMAssistantMessage, LLMToolMessage]


class AgentQueryRequest(BaseModel):
    """Request model for agent query endpoint"""
    query: str
    messages: Optional[List[LLMMessage]] = Field(
        default=None,
        description="Optional previous conversation messages in LLM format. If provided, the new query will be appended to continue the conversation.",
        examples=[None, [{"role": "user", "content": "Previous question"}]]
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "How many GPUs were used to train the original transformer?",
                "messages": None
            }
        }


class AgentQueryResponse(BaseModel):
    """Response model for agent query endpoint"""
    messages: List[Dict[str, Any]]
    
    # Pydantic v2: Preserve None values in JSON serialization
    # This ensures messages with content: null (tool_calls) are included
    model_config = ConfigDict(
        exclude_none=False,  # Don't exclude None values when serializing
    )


class Message(BaseModel):
    """Message model for conversation"""
    role: str
    content: Any


class ToolCall(BaseModel):
    """Tool call model"""
    name: str
    args: Dict[str, Any]


class ToolInfo(BaseModel):
    """Tool information model"""
    name: str
    description: str
    input_schema: Dict[str, Any]


class ToolsResponse(BaseModel):
    """Response model for tools endpoint"""
    tools: List[ToolInfo]


class RetrievePageChunksRequest(BaseModel):
    """Request model for retrieve_page_chunks endpoint"""
    queries: List[str]
    use_reranking: bool = True
    top_k_ann: Optional[int] = None
    top_k_rerank: Optional[int] = None
    filter: Optional[Dict[str, Any]] = None
    force_pdf_to_text: bool = True
    force_pdf_to_text: bool = True
    
    @field_validator('filter', mode='before')
    @classmethod
    def normalize_empty_filter(cls, v):
        """
        Normalize filter to remove invalid empty dict values.
        
        ChromaDB filter syntax:
        - Valid: {"metadata_field": "value"} or {"metadata_field": {"$eq": "value"}}
        - Invalid: {"metadata_field": {}} (empty dict as value)
        
        This validator:
        - Removes keys with empty dict values (invalid in ChromaDB)
        - Returns None if filter becomes empty after cleaning
        - Preserves valid filter structures
        """
        if v is None:
            return None
        
        if not isinstance(v, dict):
            return v
        
        # If top-level dict is empty, return None
        if len(v) == 0:
            return None
        
        # Clean filter: remove keys with empty dict values
        cleaned = {}
        for key, value in v.items():
            if isinstance(value, dict):
                # If value is an empty dict, skip this key (invalid in ChromaDB)
                # ChromaDB expects: {"key": "value"} or {"key": {"$eq": "value"}}, not {"key": {}}
                if len(value) == 0:
                    continue
                # Recursively clean nested dicts (for $and, $or operators)
                cleaned_value = cls.normalize_empty_filter(value)
                if cleaned_value is not None:
                    cleaned[key] = cleaned_value
            else:
                # Non-dict values (strings, numbers, etc.) are valid
                cleaned[key] = value
        
        # If after cleaning, the dict is empty, return None
        if len(cleaned) == 0:
            return None
        
        return cleaned


class PageChunkResult(BaseModel):
    """Single page chunk result"""
    chunk_id: str
    chunk_name: str
    score: float
    chunk_text: Optional[str] = ""


class QueryResults(BaseModel):
    """Results for a single query"""
    query: str
    chunks: List[PageChunkResult]


class RetrievePageChunksResponse(BaseModel):
    """Response model for retrieve_page_chunks endpoint"""
    results: List[QueryResults]  

