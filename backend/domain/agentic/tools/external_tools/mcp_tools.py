"""
MCP tool wrapper (external) - wraps MCP tools as BaseTool instances
"""

import logging
from typing import Dict, Any
from domain.agentic.tools.base import BaseTool
from domain.agentic.mcp.client import MCPClient
from core.exceptions import ToolExecutionError

logger = logging.getLogger(__name__)


class MCPToolAdapter(BaseTool):
    """Adapts an MCP tool from MCP server to a BaseTool instance"""
    
    def __init__(self, mcp_client: MCPClient, tool_name: str, tool_description: str, tool_schema: Dict[str, Any]):
        self.mcp_client = mcp_client
        self._name = tool_name
        self._description = tool_description
        self._input_schema = tool_schema
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def description(self) -> str:
        return self._description
    
    @property
    def input_schema(self) -> Dict[str, Any]:
        return self._input_schema
    
    async def execute(self, **kwargs) -> Any:
        """
        Execute MCP tool via MCP client.
        
        Args:
            **kwargs: Tool arguments
            
        Returns:
            Tool execution result (converted to text chunks)
            
        Raises:
            ToolExecutionError: If tool execution fails or client is not connected
        """
        if not self.mcp_client.is_connected():
            error_msg = f"MCP client is not connected. Cannot execute tool {self._name}."
            logger.error(error_msg)
            raise ToolExecutionError(error_msg)
        
        try:
            result = await self.mcp_client.call_tool(self._name, kwargs)
            return self.mcp_client.convert_result_to_text(result)
            
        except Exception as e:
            logger.error(f"Error calling MCP tool {self._name}: {e}")
            raise ToolExecutionError(f"Failed to execute MCP tool {self._name}: {e}")

