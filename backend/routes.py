"""
API endpoints (routes)
"""

from fastapi import APIRouter, HTTPException, Depends
from mcp_client import MCPClient
from models import QueryRequest
from dependencies import get_client

router = APIRouter()


@router.post("/query")
async def process_query(
    request: QueryRequest,
    mcp_client: MCPClient = Depends(get_client)
):
    """Process a query and return the response"""
    try:
        messages = await mcp_client.process_query(request.query)
        return {"messages": messages}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tools")
async def get_tools(
    mcp_client: MCPClient = Depends(get_client)
):
    """Get the list of available tools"""
    try:
        tools = await mcp_client.get_mcp_tools()
        return {
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema,
                }
                for tool in tools
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

