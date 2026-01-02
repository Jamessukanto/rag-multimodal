"""
Groq LLM client implementation
"""

import json
import logging
from typing import List, Dict, Any, Optional
from groq import Groq

from domain.agentic.llm.base import BaseLLMClient
from domain.agentic.tools.base import BaseTool
from core.exceptions import LLMError

logger = logging.getLogger(__name__)


class GroqClient(BaseLLMClient):
    """Groq LLM client implementation"""
    
    def __init__(self, model: str, max_tokens: int, api_key: Optional[str] = None):
        super().__init__(model, max_tokens)
        try:
            self.client = Groq(api_key=api_key)
        except Exception as e:
            raise LLMError(f"Failed to initialize Groq client: {e}")
    
    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Any:
        """Make a chat completion request to Groq"""
        try:
            params = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "messages": messages,
            }
            if tools:
                params["tools"] = tools
            
            response = self.client.chat.completions.create(**params)
            return response
        except Exception as e:
            logger.error(f"Error calling Groq API: {e}")
            raise LLMError(f"Groq API call failed: {e}")
    
    def extract_tool_calls(self, response: Any) -> List[Dict[str, Any]]:
        """Extract and format Groq tool calls to standard format"""
        if not response.choices:
            return []
        message = response.choices[0].message
        tool_calls = getattr(message, "tool_calls", None)
        if not tool_calls:
            return []
        
        formatted = []
        for tool_call in tool_calls:
            formatted.append({
                "id": tool_call.id,
                "name": tool_call.function.name,
                "arguments": tool_call.function.arguments,
            })
        return formatted
    
    def extract_text_content(self, response: Any) -> str:
        """Extract text content from Groq response"""
        if not response.choices:
            return ""
        message = response.choices[0].message
        return message.content or ""
    
    def has_tool_calls(self, response: Any) -> bool:
        """Check if Groq response contains tool calls"""
        if not response.choices:
            return False
        message = response.choices[0].message
        return bool(getattr(message, "tool_calls", None))
    
    def format_tool_message(
        self,
        tool_calls: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Format assistant message with tool calls for Groq (OpenAI format)"""
        return {
            "role": "assistant",
            "content": None,  # Must be None when tool_calls are present
            "tool_calls": [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": tc["arguments"],
                    },
                }
                for tc in tool_calls
            ],
        }
    
    def format_tool_result_message(
        self,
        tool_call_id: str,
        tool_name: str,
        tool_result: str
    ) -> Dict[str, Any]:
        """Format tool result message for Groq (OpenAI format)"""
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": tool_result,
        }
    
    def format_tools(self, tools: List[BaseTool]) -> List[Dict[str, Any]]:
        """
        Convert tools from BaseTool objects to OpenAI/Groq format.
        
        Groq format: [{"type": "function", "function": {...}}]
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                }
            }
            for tool in tools
        ]

