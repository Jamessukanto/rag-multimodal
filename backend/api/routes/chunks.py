"""
Chunk management API endpoints
"""

import logging
from fastapi import APIRouter, Depends, HTTPException, status
from typing import Optional

from api.dependencies import get_document_service
from api.schemas.chunks import (
    ChunkInfo,
    ChunkListResponse,
    ChunkDeleteResponse,
    ChunkDeleteAllResponse,
)
from services.document_service import DocumentService
from core.exceptions import StorageError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/chunks", tags=["chunks"])


@router.get("", response_model=ChunkListResponse)
async def list_chunks(
    document_service: DocumentService = Depends(get_document_service)
):
    """List chunks."""
    try:
        # Get all chunks by listing all documents and their chunks
        documents = await document_service.list_documents()
        all_chunks = []
        for doc in documents:
            doc_chunks = await document_service.document_sql_store.get_chunks_by_document(doc["doc_id"])
            all_chunks.extend(doc_chunks)
        chunks = all_chunks
        
        return ChunkListResponse(
            chunks=[ChunkInfo(**chunk) for chunk in chunks],
            total=len(chunks)
        )
    except StorageError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error listing chunks: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list chunks"
        )


@router.get("/{chunk_id}", response_model=ChunkInfo)
async def get_chunk(
    chunk_id: str,
    document_service: DocumentService = Depends(get_document_service)
):
    """
    Get a specific chunk by ID.
    
    Returns chunk metadata including doc_id, chunk_name, chunk_path, etc.
    """
    try:
        chunk = await document_service.document_sql_store.get_chunk(chunk_id)
        if not chunk:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Chunk {chunk_id} not found"
            )
        return ChunkInfo(**chunk)
    except HTTPException:
        raise
    except StorageError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND if "not found" in str(e).lower() else status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error getting chunk {chunk_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get chunk"
        )


@router.delete("/all", response_model=ChunkDeleteAllResponse)
async def delete_all_chunks(
    document_service: DocumentService = Depends(get_document_service)
):
    """
    Delete all chunks and their embeddings.
    
    ⚠️ DANGEROUS OPERATION ⚠️
    
    This will delete:
    - All chunks from all documents
    - All chunk files
    - All chunk embeddings from vector stores
    
    This operation cannot be undone.
    """
    try:
        # Get all documents and delete them (which cascades to chunks)
        documents = await document_service.list_documents()
        total_chunks = 0
        
        for doc in documents:
            doc_info = await document_service.get_document(doc["doc_id"])
            if doc_info.get("chunks"):
                total_chunks += len(doc_info["chunks"])
            await document_service.delete_document(doc["doc_id"])
        
        return ChunkDeleteAllResponse(
            num_chunks_deleted=total_chunks,
            status="deleted"
        )
    except Exception as e:
        logger.error(f"Unexpected error deleting all chunks: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete all chunks"
        )


# @router.delete("/{chunk_id}", response_model=ChunkDeleteResponse)
# async def delete_chunk(
#     chunk_id: str,
#     document_service: DocumentService = Depends(get_document_service)
# ):
#     """
#     Delete a chunk and all associated data.
    
#     Deletes:
#     - Chunk metadata (from database)
#     - Chunk file
#     - Single and multi vector embeddings from vector stores
#     """
#     try:
#         result = await document_service.delete_chunk(chunk_id)
#         return ChunkDeleteResponse(**result)
#     except StorageError as e:
#         raise HTTPException(
#             status_code=status.HTTP_404_NOT_FOUND if "not found" in str(e).lower() else status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=str(e)
#         )
#     except Exception as e:
#         logger.error(f"Unexpected error deleting chunk {chunk_id}: {e}", exc_info=True)
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Failed to delete chunk"
#         )

