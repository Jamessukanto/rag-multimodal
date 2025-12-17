"""
Pydantic models for request/response schemas
"""

from pydantic import BaseModel
from typing import Dict, Any


class QueryRequest(BaseModel):
    """Request model for /query endpoint"""
    query: str


class Message(BaseModel):
    """Message model for conversation"""
    role: str
    content: Any


class ToolCall(BaseModel):
    """Tool call model"""
    name: str
    args: Dict[str, Any]

