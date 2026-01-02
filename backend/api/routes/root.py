"""
Root and health check endpoints
"""

from fastapi import APIRouter

router = APIRouter(tags=["root"])

@router.get("/")
async def root():
    """API root endpoint - returns API information"""
    return {
        "name": "Agentic RAG API",
        "version": "0.1.0",
        "docs": "/docs",
        "endpoints": {
            "agent": "/api/v1/agent",
            "documents": "/api/v1/documents",
            "ingestion": "/api/v1/ingestion",
            "evaluation": "/api/v1/evaluation",
        }
    }

