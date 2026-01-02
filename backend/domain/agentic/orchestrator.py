"""
Agent orchestrator - manages conversation flow and tool calling
"""

import json
import logging
from typing import List, Dict, Any, Optional

from domain.agentic.llm.base import BaseLLMClient
from domain.agentic.tools.registry import ToolRegistry
from domain.agentic.tools.base import BaseTool
from core.exceptions import AgenticException, LLMError, ToolExecutionError

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """
    Orchestrates agentic queries with LLM and tools.
    Manages conversation state and tool using provider-agnostic logic.
    """
    
    def __init__(
        self,
        llm_client: BaseLLMClient,
        tool_registry: ToolRegistry,
    ):
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.logger = logger
        self._formatted_tools: Optional[List[Dict[str, Any]]] = None
    
    async def process_query(
        self, 
        query: str,
        messages: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Process a user query through the agent system.
        Loop until LLM returns final answer (no tool calls).
        
        Args:
            query: New user query to process.
            messages: Optional previous conversation messages in LLM format.
                     Expected format matches what the LLM API expects:
                     - {"role": "user", "content": "..."}
                     - {"role": "assistant", "content": "..."} or {"role": "assistant", "content": None, "tool_calls": [...]}
                     - {"role": "tool", "tool_call_id": "...", "name": "...", "content": "..."}
        
        Returns:
            Complete conversation history in LLM format, including the new query and response.
        """
        try:
            self.logger.info(f"Processing query: {query}")

            messages = messages.copy() if messages else []
            messages.append({"role": "user", "content": query})
            
            # Cached. Lazy as internal tools are not initialised upon setup
            tools = self._get_formatted_tools()
            
            while True:
                response = await self.llm_client.chat_completion(
                    messages=messages,
                    tools=tools,
                )
                
                # LLM returned no tool calls → final text answer
                if not self.llm_client.has_tool_calls(response):
                    final_answer = self.llm_client.extract_text_content(response)
                    messages.append({"role": "assistant", "content": final_answer})
                    break
                
                # LLM returned tool calls → execute and continue loop
                extracted_tool_calls = self.llm_client.extract_tool_calls(response)
                messages.append(self.llm_client.format_tool_message(extracted_tool_calls))
                
                tool_results = await self._execute_tools(extracted_tool_calls)
                for result in tool_results:
                    tool_call = result["tool_call"]
                    messages.append(
                        self.llm_client.format_tool_result_message(
                            tool_call_id=tool_call["id"],
                            tool_name=tool_call["name"],
                            tool_result=result["result_text"]
                        )
                    )

            # Return LLM format directly (no conversion needed)
            return messages

        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            raise AgenticException(f"Failed to process query: {e}")
    
    def list_tools(self) -> List[BaseTool]:
        """List all available tools (MCP + RAG tools)."""
        return self.tool_registry.get_all_tools()
    
    def _get_formatted_tools(self) -> List[Dict[str, Any]]:
        """
        Get formatted tools for LLM (cached after first call).
        Tools don't change at runtime, so we cache the formatted version.
        """
        if self._formatted_tools is None:
            all_tools = self.tool_registry.get_all_tools()
            self._formatted_tools = self.llm_client.format_tools(all_tools)
        return self._formatted_tools
    
    
    async def _execute_tools(
        self,
        formatted_tool_calls: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Execute all tool calls and return results."""
        tool_results = []

        for tool_call in formatted_tool_calls:
            tool_name = tool_call["name"]
            tool_args = json.loads(tool_call["arguments"])
            tool_result_text, text_chunks = await self._execute_single_tool(
                tool_name, tool_args
            )
            tool_results.append({
                "tool_call": tool_call,
                "result_text": tool_result_text,
                "text_chunks": text_chunks,
            })
        return tool_results
    
    async def _execute_single_tool(
        self,
        tool_name: str,
        tool_args: Dict[str, Any]
    ) -> tuple[str, List[str]]:
        """Execute a single tool and return the result."""
        self.logger.info(f"Calling tool {tool_name} with args {tool_args}")
        
        try:
            # Execute tool via registry (handles both MCP and RAG tools)
            result = await self.tool_registry.execute_tool(tool_name, tool_args)
            self.logger.info(f"Tool {tool_name} result: {result}...")
            
            # Convert result to text chunks
            # Both MCP and RAG tools return strings
            if isinstance(result, str):
                text_chunks = [result]
                tool_result_text = result
            else:
                tool_result_text = str(result)
                text_chunks = [tool_result_text]
            
            return tool_result_text, text_chunks
        except Exception as e:
            self.logger.error(f"Error calling tool {tool_name}: {e}")
            raise ToolExecutionError(f"Failed to execute tool {tool_name}: {e}")
    
    