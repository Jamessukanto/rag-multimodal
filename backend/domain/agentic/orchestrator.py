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
                    self.logger.info(f"Final answer: {final_answer}")
                    break
                
                # Extract tool calls => Execute => Append to messages
                tool_calls = self.llm_client.extract_tool_calls(response)
                tool_results = await self._execute_tools(tool_calls)

                messages.append(
                    self.llm_client.format_tool_message(tool_calls)
                )
                messages.extend([                        
                    self.llm_client.format_tool_result_message(
                        tool_call_id=tc["id"],
                        tool_name=tc["name"],
                        tool_result=res
                    ) for tc, res in zip(tool_calls, tool_results)
                ])

            return messages

        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            raise AgenticException(f"Failed to process query: {e}")
    

    def list_tools(self) -> List[BaseTool]:
        """List all available tools (MCP + RAG tools)."""
        return self.tool_registry.get_all_tools()
    

    def _get_formatted_tools(self) -> List[Dict[str, Any]]:
        """Get formatted tools for LLM (cached after first call)."""

        if self._formatted_tools is None:
            all_tools = self.tool_registry.get_all_tools()
            self._formatted_tools = self.llm_client.format_tools(all_tools)

        return self._formatted_tools
    
    
    async def _execute_tools(
        self,
        tool_calls: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Execute all tool calls and return results."""
        
        tool_results = []

        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_args = json.loads(tool_call["arguments"])

            result = await self.tool_registry.execute_tool(tool_name, tool_args)
            tool_results.append(str(result))

        return tool_results
    
