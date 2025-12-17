"""
Shared dependencies
"""

from fastapi import Request
from mcp_client import MCPClient


def get_client(request: Request) -> MCPClient:
    """Dependency to get MCP client from app state"""
    return request.app.state.client

