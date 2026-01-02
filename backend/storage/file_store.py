"""
PDF file storage abstraction
"""

import logging
from pathlib import Path
from typing import Optional
from core.config import settings
from core.exceptions import StorageError
from storage.base import BaseFileStore

logger = logging.getLogger(__name__)


class FileStore(BaseFileStore):
    """
    File storage for PDF documents.
    Supports local file system (development) and object storage (production).
    """
    
    def __init__(self, storage_type: str = None, base_path: Path = None):
        """
        Initialize file store.
        
        Args:
            storage_type: "filesystem" (default), "s3", or "minio"
            base_path: Base path for file system storage (defaults to settings.documents_dir)
        """
        self.storage_type = storage_type or settings.file_storage_type
        self.base_path = base_path or settings.documents_dir
        
        if self.storage_type == "filesystem":
            self.base_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"FileStore initialized with filesystem storage at {self.base_path}")
        else:
            # TODO: Implement S3/MinIO support
            raise NotImplementedError(f"Storage type {self.storage_type} not yet implemented")
    
    def get_file_path(self, doc_id: str) -> Path:
        return self.base_path / f"{doc_id}.pdf"

    async def save_file(self, doc_id: str, file_content_bytes: bytes) -> Path:
        try:
            file_path = self.get_file_path(doc_id)
            file_path.write_bytes(file_content_bytes)
            logger.info(f"Saved file for doc_id {doc_id} to {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Error saving file for doc_id {doc_id}: {e}")
            raise StorageError(f"Failed to save file: {e}")
    
    async def get_file(self, doc_id: str) -> Optional[bytes]:
        try:
            file_path = self.get_file_path(doc_id)
            if not file_path.exists():
                return None
            return file_path.read_bytes()
        except Exception as e:
            logger.error(f"Error getting file for doc_id {doc_id}: {e}")
            raise StorageError(f"Failed to get file: {e}")
    
    async def delete_file(self, doc_id: str) -> None:
        try:
            file_path = self.get_file_path(doc_id)
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Deleted file for doc_id {doc_id}")
        except Exception as e:
            logger.error(f"Error deleting file for doc_id {doc_id}: {e}")
            raise StorageError(f"Failed to delete file: {e}")
    
    def file_exists(self, doc_id: str) -> bool:
        file_path = self.get_file_path(doc_id)
        return file_path.exists()
    


