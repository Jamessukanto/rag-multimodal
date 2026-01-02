"""
Multi-vector store implementation (file-based for dev)
"""

import logging
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from storage.base import BaseMultiVectorStore
from core.config import settings
from core.exceptions import StorageError

logger = logging.getLogger(__name__)


class MultiVectorStore(BaseMultiVectorStore):
    """Multi-vector store using file-based storage"""
    
    def __init__(
        self,
        store_path=None,
    ):
        store_path = store_path or settings.multi_vector_store_path
        if isinstance(store_path, str):
            store_path = Path(store_path)
        
        # Resolve relative paths relative to backend directory
        if not store_path.is_absolute():
            # Get backend directory (parent of storage directory)
            backend_dir = Path(__file__).parent.parent
            store_path = (backend_dir / store_path).resolve()

        self.store_path = store_path
        self.store_path.mkdir(parents=True, exist_ok=True)
        self.index_file = self.store_path / "multi_vector_index.pkl"
        self._index: Dict[str, List[List[float]]] = self._load_index()

        logger.info(f"MultiVectorStore initialized at: {self.store_path}")

    def _load_index(self) -> Dict[str, List[List[float]]]:
        """Load index from disk"""
        if self.index_file.exists():
            try:
                with open(self.index_file, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load multi-vector index: {e}")
        return {}
    
    def _save_index(self):
        """Save index to disk"""
        try:
            with open(self.index_file, "wb") as f:
                pickle.dump(self._index, f)
        except Exception as e:
            logger.error(f"Failed to save multi-vector index: {e}")
            raise StorageError(f"Failed to save index: {e}")
    
    async def add(
        self,
        chunk_id: str,
        embeddings: List[List[float]]
    ) -> None:
        """Add multi-vectors to the store"""
        try:
            self._index[chunk_id] = embeddings
            self._save_index()
        except Exception as e:
            logger.error(f"Error adding multi-vectors {chunk_id}: {e}")
            raise StorageError(f"Failed to add multi-vectors: {e}")
    
    async def get(self, chunk_id: str) -> Optional[List[List[float]]]:
        """Get multi-vectors for a chunk"""
        return self._index.get(chunk_id)
    
    async def batch_get(
        self,
        chunk_ids: List[str]
    ) -> Dict[str, List[List[float]]]:
        """Get multi-vectors for multiple chunks"""
        return {cid: self._index.get(cid) for cid in chunk_ids if cid in self._index}
    
    async def delete(self, chunk_id: str) -> None:
        """Delete multi-vectors from the store"""
        try:
            if chunk_id in self._index:
                del self._index[chunk_id]
                self._save_index()
        except Exception as e:
            logger.error(f"Error deleting multi-vectors {chunk_id}: {e}")
            raise StorageError(f"Failed to delete multi-vectors: {e}")

