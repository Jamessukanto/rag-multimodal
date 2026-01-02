"""
Abstract tool interface
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List


class BaseTool(ABC):
    """Abstract base class for all tools"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name"""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description"""
        pass
    
    @property
    @abstractmethod
    def input_schema(self) -> Dict[str, Any]:
        """JSON schema for tool inputs"""
        pass
    
    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """
        Execute the tool.
        
        Args:
            **kwargs: Tool arguments
            
        Returns:
            Tool execution result
        """
        pass

