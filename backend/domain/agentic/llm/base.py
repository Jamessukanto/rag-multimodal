"""
Abstract base class for LLM clients
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from agentic.tools.base import BaseTool

from core.exceptions import LLMError


class BaseLLMClient(ABC):
    """Abstract base class for all LLM clients"""
    
    def __init__(self, model: str, max_tokens: int):
        self.model = model
        self.max_tokens = max_tokens
    
    @abstractmethod
    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Any:
        """
        Make a chat completion request to the LLM.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            tools: Optional list of tool definitions

        Returns:
            LLM response object (provider-specific)

        Raises:
            LLMError: If the request fails
        """
        pass
    
    @abstractmethod
    def extract_tool_calls(self, response: Any) -> List[Dict[str, Any]]:
        """
        Extract and format tool calls from LLM response to a standard format.
        
        Args:
            response: LLM response object (provider-specific)
            
        Returns:
            List of tool call dictionaries with 'id', 'name', 'arguments'
        """
        pass
    
    @abstractmethod
    def extract_text_content(self, response: Any) -> str:
        """
        Extract text content from LLM response.
        
        Args:
            response: LLM response object (provider-specific)
            
        Returns:
            Text content string
        """
        pass
    
    @abstractmethod
    def has_tool_calls(self, response: Any) -> bool:
        """
        Check if LLM response contains tool calls.
        
        Args:
            response: LLM response object (provider-specific)
            
        Returns:
            True if response contains tool calls, False otherwise
        """
        pass
    
    @abstractmethod
    def format_tool_message(
        self,
        tool_calls: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Format an assistant message with tool calls for the LLM's message format.
        
        Args:
            tool_calls: List of formatted tool calls (from extract_tool_calls)
            
        Returns:
            Message dictionary in the provider's format
        """
        pass
    
    @abstractmethod
    def format_tool_result_message(
        self,
        tool_call_id: str,
        tool_name: str,
        tool_result: str
    ) -> Dict[str, Any]:
        """
        Format a tool result message for the LLM's message format.
        
        Args:
            tool_call_id: ID of the tool call
            tool_name: Name of the tool
            tool_result: Result text from the tool
            
        Returns:
            Message dictionary in the provider's format
        """
        pass
    
    @abstractmethod
    def format_tools(self, tools: List["BaseTool"]) -> List[Dict[str, Any]]:
        """
        Convert tools from BaseTool objects to provider-specific format.
        
        Args:
            tools: List of BaseTool instances
            
        Returns:
            List of tools in provider-specific format
        """
        pass

