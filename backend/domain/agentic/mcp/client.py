"""
MCP client for connecting to external MCP servers
Extracted from mcp_client.py - handles MCP protocol communication only
"""

import logging
import os
import traceback
from typing import Optional, List, Dict, Any
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from core.exceptions import MCPConnectionError
from core.config import settings

logger = logging.getLogger(__name__)


class MCPClient:
    """
    Client for connecting to external MCP servers via stdio.
    Handles connection, tool discovery, and tool execution.
    """
    
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.tools: List[Dict[str, Any]] = []
        self.logger = logger
    
    async def connect_to_server(self, server_script_path: Optional[str] = None) -> bool:
        server_script_path = server_script_path or settings.mcp_server_script_path
        
        try:
            is_python = server_script_path.endswith(".py")
            if not is_python:
            # if not (is_python or is_js):
                raise ValueError("Server script must be a .py file for now")
            
            if is_python:
                # Use uv run to ensure MCP server has its dependencies
                server_dir = os.path.dirname(os.path.abspath(server_script_path))
                args = [
                    "--directory", server_dir,
                    "run", "python", os.path.basename(server_script_path)
                ]
                server_params = StdioServerParameters(
                    command="uv", args=args, env=None
                )
                
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            self.stdio, self.write = stdio_transport
            self.session = await self.exit_stack.enter_async_context(
                ClientSession(self.stdio, self.write)
            )
            
            await self.session.initialize()
            self.logger.info("Connected to MCP server")
            
            # Discover and format tools
            mcp_tools = await self.get_mcp_tools()
            self.tools = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema,
                    }
                }
                for tool in mcp_tools
            ]
            
            self.logger.info(
                f"Available tools: {[tool['function']['name'] for tool in self.tools]}"
            )
            self.logger.debug(f"Tools format: {self.tools[0] if self.tools else 'No tools'}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error connecting to MCP server: {e}")
            traceback.print_exc()
            raise MCPConnectionError(f"Failed to connect to MCP server: {e}")
    
    async def get_mcp_tools(self) -> List[Any]:
        if not self.session:
            raise MCPConnectionError("Not connected to MCP server. Call connect_to_server() first.")
        
        try:
            response = await self.session.list_tools()
            return response.tools
        except Exception as e:
            self.logger.error(f"Error getting MCP tools: {e}")
            raise MCPConnectionError(f"Failed to get MCP tools: {e}")
    
    def is_connected(self) -> bool:
        """Check if MCP client is connected to server."""
        return self.session is not None
    
    async def call_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Any:
        if not self.is_connected():
            raise MCPConnectionError("Not connected to MCP server. Call connect_to_server() first.")
        
        try:
            result = await self.session.call_tool(tool_name, tool_args)
            return result
        except Exception as e:
            self.logger.error(f"Error calling MCP tool {tool_name}: {e}")
            raise
    
    def convert_result_to_text(self, result: Any) -> str:
        """Convert MCP tool result to plain text."""
        if not result or not hasattr(result, "content"):
            return ""
        
        text_chunks = [
            getattr(c, "text", str(c))
            for c in (result.content or [])
        ]
        return "\n".join(text_chunks)
    
    async def cleanup(self):
        try:
            await self.exit_stack.aclose()
            self.logger.info("Disconnected from MCP server")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            traceback.print_exc()
            raise

