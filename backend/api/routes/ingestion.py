"""
Ingestion endpoints - handles document splitting and embedding
"""

import logging
from fastapi import APIRouter, HTTPException, Depends, status

from api.dependencies import get_ingestion_service
from api.schemas.ingestion import IngestAllResponse
from services.ingestion_service import IngestionService
from core.exceptions import IngestionError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/ingestion", tags=["ingestion"])


@router.post("/ingest_all", response_model=IngestAllResponse, status_code=status.HTTP_200_OK)
async def ingest_all(
    ingestion_service: IngestionService = Depends(get_ingestion_service)
):
    """
    Ingest all unprocessed documents.
    
    Processes all documents with status "uploaded" or "error":
    - Splits PDFs into page chunks
    - Stores the split chunks and metadata separately
    - Generates embeddings
    
    Returns a summary of processed documents and any failures.
    """
    try:
        logger.info("Starting ingestion of all unprocessed documents")
        result = await ingestion_service.ingest_unprocessed_documents()
        return IngestAllResponse(**result)
    except IngestionError as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error during ingestion: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest documents: {str(e)}"
        )