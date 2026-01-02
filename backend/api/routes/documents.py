"""
Document management API endpoints
"""

import logging
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status, Path
from typing import List

from api.dependencies import get_document_service
from api.schemas.documents import (
    DocumentUploadResponse,
    DocumentUploadsResponse,
    DocumentStatusResponse,
    DocumentInfo,
    DocumentListResponse,
    DocumentDeleteResponse,
    DocumentDeleteAllResponse,
)
from services.document_service import DocumentService
from core.exceptions import StorageError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    document_service: DocumentService = Depends(get_document_service)
):
    """
    List all documents.
    
    Returns list of documents with metadata.
    """
    try:
        documents = await document_service.list_documents()
        return DocumentListResponse(
            documents=[DocumentInfo(**doc) for doc in documents],
            total=len(documents)
        )
    except StorageError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error listing documents: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list documents"
        )


@router.delete("/all", response_model=DocumentDeleteAllResponse)
async def delete_all_documents(
    document_service: DocumentService = Depends(get_document_service)
):
    """
    ⚠️ DANGEROUS OPERATION ⚠️  Delete all documents and all associated data.
    
    This will delete:
    - All documents and chunk metadata
    - All document files
    - All chunk files
    - All embeddings from vector stores
    
    This operation cannot be undone.
    """
    try:
        documents = await document_service.list_documents()
        logger.info(f"Found {len(documents)} documents to delete")
        
        deleted_count = 0
        failed_count = 0
        
        for doc in documents:
            doc_id = doc.get("doc_id")
            if not doc_id:
                logger.warning(f"Skipping document with missing doc_id: {doc}")
                failed_count += 1
                continue
                
            try:
                await document_service.delete_document(doc_id)
                deleted_count += 1
                logger.info(f"Successfully deleted document {doc_id}")
            except Exception as e:
                logger.warning(f"Failed to delete document {doc_id}: {e}")
                failed_count += 1
        
        logger.info(f"Delete all completed: {deleted_count} successful, {failed_count} failed out of {len(documents)} total")
        
        return DocumentDeleteAllResponse(
            num_documents_deleted=deleted_count,
            status="deleted"
        )
    except Exception as e:
        logger.error(f"Unexpected error deleting all documents: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete all documents"
        )


@router.get("/{doc_id}", response_model=DocumentInfo)
async def get_document(
    doc_id: str = Path(..., description="Document ID to retrieve"),
    document_service: DocumentService = Depends(get_document_service)
):
    """Get document metadata by document ID."""
    try:
        result = await document_service.get_document(doc_id)
        return DocumentInfo(**result)
    except StorageError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND if "not found" in str(e).lower() else status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error getting document {doc_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get document"
        )


@router.delete("/{doc_id}", response_model=DocumentDeleteResponse)
async def delete_document(
    doc_id: str = Path(..., description="Document ID to delete"),
    document_service: DocumentService = Depends(get_document_service)
):
    """
    Delete a document and all associated data.
    
    Deletes:
    - Document and chunk metadata (cascade delete in database)
    - Document file
    - Chunk files
    - Single and multi vector embeddings from vector stores
    """
    try:
        result = await document_service.delete_document(doc_id)
        return DocumentDeleteResponse(**result)
    except StorageError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND if "not found" in str(e).lower() else status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error deleting document {doc_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete document"
        )


@router.post("/uploads", response_model=DocumentUploadsResponse, status_code=status.HTTP_201_CREATED)
async def upload_documents(
    files: List[UploadFile] = File(..., description="PDF files to upload (one or more)"),
    document_service: DocumentService = Depends(get_document_service)
):
    """Upload one or more PDF documents."""
    try:
        if not files:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one file is required"
            )
        
        uploaded_documents = []
        failed_count = 0
        
        for file in files:
            try:
                # Validate file
                if not file.filename or not file.filename.endswith('.pdf'):
                    logger.warning(f"Skipping invalid file: {file.filename}")
                    failed_count += 1
                    continue
                
                file_content_bytes = await file.read()
                result = await document_service.upload_document(
                    file_content_bytes=file_content_bytes,
                    doc_name=file.filename
                )
                
                uploaded_documents.append(DocumentUploadResponse(**result))
                
            except StorageError as e:
                logger.error(f"StorageError uploading {file.filename}: {e}")
                failed_count += 1
            except Exception as e:
                logger.error(f"Unexpected error uploading {file.filename}: {e}", exc_info=True)
                failed_count += 1
        
        # If all files failed, return error
        if len(uploaded_documents) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"All {len(files)} file(s) failed to upload"
            )
        
        return DocumentUploadsResponse(
            documents=uploaded_documents,
            total=len(files),
            successful=len(uploaded_documents),
            failed=failed_count
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in upload_documents: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload documents: {type(e).__name__}: {str(e)}"
        )





