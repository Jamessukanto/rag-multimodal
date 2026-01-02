"""
Vector storage layer
"""

from storage.base import BaseSingleVectorStore, BaseMultiVectorStore, BaseDocumentSQLStore, BaseFileStore
from storage.document_sql_store import DocumentSQLStore
from storage.file_store import FileStore

# Optional imports - only available if chromadb is installed
try:
    from storage.single_vector_store import SingleVectorStore
    from storage.multi_vector_store import MultiVectorStore
except ImportError:
    SingleVectorStore = None
    MultiVectorStore = None

__all__ = [
    "BaseSingleVectorStore",
    "BaseMultiVectorStore",
    "BaseDocumentSQLStore",
    "BaseFileStore",
    "DocumentSQLStore",
    "FileStore",
    "SingleVectorStore",  # May be None if chromadb not installed
    "MultiVectorStore",   # May be None if chromadb not installed
]

