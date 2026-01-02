"""
App setup, middleware, lifespan
"""

import logging
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from core.config import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
from core.startup import (
    initialize_agentic_system,
    initialize_rag_system,
    cleanup_agentic_system,
    cleanup_rag_system,
    raise_startup_error,
)
from api.routes.root import router as root_router
from api.routes.agent import router as agent_router
from api.routes.documents import router as documents_router
from api.routes.chunks import router as chunks_router
from api.routes.ingestion import router as ingestion_router
from api.routes.evaluation import router as evaluation_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown logic.
    """
    try:
        await initialize_agentic_system(app)
        await initialize_rag_system(app)
        yield

    except HTTPException:
        # Re-raise expected HTTPExceptions from raise_startup_error
        raise

    except Exception as e:
        logger.error(f"Unexpected error during lifespan: {e}", exc_info=True)
        raise_startup_error("Error during application startup", e)
    
    # Cleanup persistent connections
    finally:
        await cleanup_rag_system(app)
        await cleanup_agentic_system(app)


app = FastAPI(
    title="Agentic RAG API",
    description="Multimodal RAG system with MCP tool integration",
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Include routes
app.include_router(root_router)
app.include_router(agent_router)
app.include_router(documents_router)
app.include_router(chunks_router)
app.include_router(ingestion_router)
app.include_router(evaluation_router)



if __name__ == "__main__":
    import uvicorn
    # Use import string for reload to work properly
    if settings.environment == "development":
        uvicorn.run(
            "main:app",  # Import string required for reload
            host=settings.api_host, 
            port=settings.api_port,
            reload=True
        )
    else:
        uvicorn.run(
            app,  # Can use app object directly in production
            host=settings.api_host, 
            port=settings.api_port
        )
