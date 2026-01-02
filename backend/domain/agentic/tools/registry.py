"""
Tool registry - central registry for all tools (MCP + RAG)
"""

import logging
from typing import List, Dict, Any, Optional, Union
from domain.agentic.tools.base import BaseTool
from domain.agentic.tools.external_tools.mcp_tools import MCPToolAdapter
from domain.agentic.tools.internal_tools.retrieval_tool import RetrieveDocumentsTool
from domain.agentic.mcp.client import MCPClient
from services.base import BaseService
from services.retrieval_service import RetrievalService
from core.exceptions import ToolExecutionError

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Central tool registry that manages all tools (MCP + RAG tools)."""
    
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self.logger = logger
    
    def _register_tool(self, tool: BaseTool):
        """Register a tool"""
        if tool.name in self._tools:
            self.logger.warning(f"Overwriting registered tool {tool.name}.")
        self._tools[tool.name] = tool
        self.logger.info(f"Registered tool: {tool.name}")
    
    def register_internal_tools(self, service: BaseService):
        """Register internal tools from a service."""
        tool = None
        if isinstance(service, RetrievalService):
            tool = RetrieveDocumentsTool(service)

        if tool is None:
            self.logger.warning(
                f"No tool for service {service.__class__.__name__} found."
            )
            return

        self._register_tool(tool)
        self.logger.info(f"Registered tool from {service.__class__.__name__}: {tool.name}")
    
    def register_external_tools(self, mcp_client: MCPClient, tool_names: Optional[List[str]] = []):
        """Register external (MCP) tools."""
        mcp_tools = mcp_client.tools
        
        # Filter provided tool names
        if tool_names:
            tool_names_set = set(tool_names)
            
            # Validate that all requested tools exist
            available_tool_names = {tool["function"]["name"] for tool in mcp_tools}
            missing_tools = tool_names_set - available_tool_names
            if missing_tools:
                raise ValueError(f"Requested tools not in MCP server: {missing_tools}.")
            mcp_tools = [
                tool for tool in mcp_tools
                if tool["function"]["name"] in tool_names_set
            ]

        self.logger.info(f"Registering MCP tools: {[tool['function']['name'] for tool in mcp_tools]}")
        
        for mcp_tool in mcp_tools:
            wrapped_tool = MCPToolAdapter(
                mcp_client=mcp_client,
                tool_name=mcp_tool["function"]["name"],
                tool_description=mcp_tool["function"]["description"],
                tool_schema=mcp_tool["function"]["parameters"]
            )
            self._register_tool(wrapped_tool)
    
    
    def get_all_tools(self) -> List[BaseTool]:
        """Get all registered tools"""
        return list(self._tools.values())
    

    async def execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Any:
        """Execute a tool by name."""
        tool = self._tools.get(tool_name)
        if not tool:
            raise ToolExecutionError(f"Tool {tool_name} not found")
        
        return await tool.execute(**tool_args)

