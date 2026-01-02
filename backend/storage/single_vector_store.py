"""
Single vector store implementation using ChromaDB.
Supports embedded (dev) and cloud (production) deployment modes.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    chromadb = None
    ChromaSettings = None

from storage.base import BaseSingleVectorStore
from core.config import settings
from core.exceptions import StorageError

logger = logging.getLogger(__name__)


class SingleVectorStore(BaseSingleVectorStore):
    """
    Single vector store using ChromaDB with flexible deployment modes.
    
    The deployment mode is determined by settings.vector_store_backend:
    - "chromadb_embedded": Local persistent storage (default for dev)
    - "chromadb_cloud": Connect to ChromaDB Cloud (managed service)
    
    To switch modes, set the VECTOR_STORE_BACKEND environment variable.
    """
    
    def __init__(
        self,
        backend_type: Optional[str] = None,
        collection_name: Optional[str] = None
    ):
        """
        Initialize ChromaDB client based on configured backend mode.
        
        Args:
            backend_type: Backend type ("chromadb_embedded" or "chromadb_cloud").
                         If None, uses settings.vector_store_backend.
            collection_name: Name of the collection. If None, uses settings.vector_store_collection_name.
        
        Raises:
            ImportError: If chromadb is not installed.
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError(
                "ChromaDB is not installed. Install it with: uv pip install chromadb"
            )
        
        backend_type = backend_type or settings.vector_store_backend
        collection_name = collection_name or settings.vector_store_collection_name
        
        if backend_type == "chromadb_embedded":
            self._init_embedded()
        elif backend_type == "chromadb_cloud":
            self._init_cloud()
        else:
            raise ValueError(
                f"Unsupported vector store backend: {backend_type}. "
                f"Supported: chromadb_embedded, chromadb_cloud"
            )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info(f"Initialized SingleVectorStore with backend: {backend_type}, collection: {collection_name}")
    
    def _init_embedded(self, store_path: Optional[Path] = None):
        """Initialize ChromaDB embedded mode (local persistent storage)"""
        store_path = store_path or settings.single_vector_store_path
        
        # Convert to Path if it's a string
        if isinstance(store_path, str):
            store_path = Path(store_path)
        
        # Resolve relative paths relative to backend directory
        if not store_path.is_absolute():
            backend_dir = Path(__file__).parent.parent
            store_path = (backend_dir / store_path).resolve()
        
        # Create directory
        store_path.mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=str(store_path),
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        
        logger.info(f"ChromaDB embedded store initialized at: {store_path}")
    
    def _init_cloud(self, api_key: str = None, tenant: str = None, database: str = None):
        """Initialize ChromaDB Cloud mode (managed service)"""
        api_key = api_key or settings.chromadb_cloud_api_key
        tenant = tenant or settings.chromadb_cloud_tenant
        database = database or settings.chromadb_cloud_database
        
        if not api_key:
            raise ValueError(
                "ChromaDB Cloud requires api_key. Set CHROMADB_CLOUD_API_KEY environment variable."
            )
        if not tenant:
            raise ValueError(
                "ChromaDB Cloud requires tenant. Set CHROMADB_CLOUD_TENANT environment variable."
            )
        
        # Note: ChromaDB Cloud client may have different API - adjust if needed
        try:
            self.client = chromadb.CloudClient(
                tenant=tenant,
                database=database,
                api_key=api_key,
                settings=ChromaSettings(anonymized_telemetry=False)
            )
        except AttributeError:
            # Fallback if CloudClient doesn't exist in this version
            raise ValueError(
                "ChromaDB Cloud client not available. "
                "Ensure you're using a version of chromadb that supports CloudClient."
            )
    
    async def add(
        self,
        chunk_id: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a vector to the store"""
        try:
            self.collection.add(
                ids=[chunk_id],
                embeddings=[embedding],
                metadatas=[metadata or {}]
            )
        except Exception as e:
            logger.error(f"Error adding vector {chunk_id}: {e}")
            raise StorageError(f"Failed to add vector: {e}")
    
    async def query(
        self,
        query_vectors: List[List[float]],
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[List[Dict[str, Any]]]:
        """
        Query the store with multiple query vectors.
        
        Args:
            query_vectors: List of query embedding vectors
            top_k: Number of results to return per query
            filter: Optional metadata filter (must not be empty dict)
            
        Returns:
            List of result lists, one per query vector. Each inner list contains dicts with:
            - 'chunk_id': str - Unique chunk identifier
            - 'score': float - Similarity score (higher is better)
            - 'metadata': Dict[str, Any] - Chunk metadata
        """
        try:
            if not query_vectors:
                raise ValueError("query_vectors list must not be empty")
            
            # Normalize empty filter to None (ChromaDB doesn't accept empty dict)
            where_clause = None if (isinstance(filter, dict) and len(filter) == 0) else filter
            
            results = self.collection.query(
                query_embeddings=query_vectors,
                n_results=top_k,
                where=where_clause
            )
            
            # Format results for each query
            all_formatted_results = []
            for query_idx in range(len(query_vectors)):
                formatted_results = []
                if results["ids"] and len(results["ids"]) > query_idx and len(results["ids"][query_idx]) > 0:
                    for i, chunk_id in enumerate(results["ids"][query_idx]):
                        formatted_results.append({
                            "chunk_id": chunk_id,
                            "score": 1.0 - results["distances"][query_idx][i],  # Convert distance to similarity
                            "metadata": results["metadatas"][query_idx][i] if results["metadatas"] and len(results["metadatas"]) > query_idx else {}
                        })
                all_formatted_results.append(formatted_results)
            
            return all_formatted_results

        except Exception as e:
            logger.error(f"Error querying vectors: {e}")
            raise StorageError(f"Failed to query vectors: {e}")
    

    async def delete(self, chunk_id: str) -> None:
        """Delete a vector from the store"""
        try:
            self.collection.delete(ids=[chunk_id])
        except Exception as e:
            logger.error(f"Error deleting vector {chunk_id}: {e}")
            raise StorageError(f"Failed to delete vector: {e}")
    
    
    async def update(
        self,
        chunk_id: str,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Update a vector in the store"""
        try:
            if embedding:
                self.collection.update(
                    ids=[chunk_id],
                    embeddings=[embedding],
                    metadatas=[metadata] if metadata else None
                )
            elif metadata:
                self.collection.update(
                    ids=[chunk_id],
                    metadatas=[metadata]
                )
        except Exception as e:
            logger.error(f"Error updating vector {chunk_id}: {e}")
            raise StorageError(f"Failed to update vector: {e}")

