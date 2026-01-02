"""
Custom exception hierarchy for the application
"""


class RAGException(Exception):
    """Base exception for RAG-related errors"""
    pass


class IngestionError(RAGException):
    """Error during document ingestion"""
    pass


class EmbeddingError(RAGException):
    """Error during embedding generation"""
    pass


class StorageError(RAGException):
    """Error during storage operations"""
    pass


class RetrievalError(RAGException):
    """Error during retrieval operations"""
    pass


class EvaluationError(RAGException):
    """Error during evaluation"""
    pass


class AgenticException(Exception):
    """Base exception for agentic system errors"""
    pass


class LLMError(AgenticException):
    """Error during LLM operations"""
    pass


class MCPConnectionError(AgenticException):
    """Error connecting to MCP server"""
    pass


class ToolExecutionError(AgenticException):
    """Error executing a tool"""
    pass

