"""
Async Jina API client with connection pooling, retry logic, and rate limiting
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Literal, Tuple
import time
import httpx
import base64
from core.config import settings
from core.exceptions import EmbeddingError
from utils.retry import retry_with_backoff
from domain.rag.embedding.types import EmbeddingResult, SingleVectorEmbedding, MultiVectorEmbedding

logger = logging.getLogger(__name__)

# Allowed task types for Jina Embedding API
TaskType = Literal["retrieval.query", "retrieval.passage"]


class JinaEmbeddingClient:
    """Async client for Jina Embedding API"""
    
    def __init__(
        self,
        task: TaskType = "retrieval.query",
        api_key: str = None,
        api_url: str = None,
        model: str = None,
        timeout: int = None,
        max_retries: int = None,
        rate_limit: int = None
    ):
        if task not in ("retrieval.query", "retrieval.passage"):
            raise ValueError(
                f"Invalid task: {task}. Must be one of: 'retrieval.query', 'retrieval.passage'"
            )
        
        self.task = task
        self.api_key = api_key or settings.jina_api_key
        if not self.api_key:
            raise EmbeddingError("Jina API key not set. Set JINA_API_KEY environment variable or pass api_key parameter.")
            
        self.api_url = api_url or settings.jina_api_url
        self.model = model or settings.jina_model
        self.timeout = timeout or settings.jina_timeout
        self.max_retries = max_retries or settings.jina_max_retries
        self.rate_limit = rate_limit or settings.jina_rate_limit

        # Connection pool
        self._client: Optional[httpx.AsyncClient] = None
        self._rate_limiter = asyncio.Semaphore(self.rate_limit)
        self._last_request_time = 0.0
        self._min_request_interval = 1.0 / self.rate_limit
    

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client"""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }
            )
        return self._client
    

    async def _rate_limit(self):
        """Rate limiting"""
        async with self._rate_limiter:
            current_time = asyncio.get_event_loop().time()
            time_since_last = current_time - self._last_request_time
            if time_since_last < self._min_request_interval:
                await asyncio.sleep(self._min_request_interval - time_since_last)
            self._last_request_time = asyncio.get_event_loop().time()
    

    def _build_input_payload(self, embedable: Dict[str, Any], id: str) -> Tuple[Any, Optional[str]]:
        """Build input payload based on task type."""

        if self.task == "retrieval.passage":
            if "pdf" not in embedable:
                raise EmbeddingError(f"pdf not found in embedable for id: {id}")
            input_payload = embedable["pdf"]
            text = None  # PDF embeddings don't have text
        elif self.task == "retrieval.query":
            if "text" not in embedable:
                raise EmbeddingError(f"text not found in embedable for id: {id}")
            input_payload = embedable["text"]
            text = embedable["text"]  # Query embeddings have text
        else:
            raise EmbeddingError(f"Invalid task: {self.task}")
        
        return input_payload, text
    

    def _build_embedding_result(
        self,
        id: str,
        text: Optional[str],
        sv_data_resp: Dict[str, Any],
        mv_data_resp: Dict[str, Any]
    ) -> EmbeddingResult:
        """Build EmbeddingResult from API responses."""
        single_vector = SingleVectorEmbedding(
            id=id,
            embedding=sv_data_resp['data'][0]["embedding"],    # (2048,)
            text=text,
            model_embed=self.model + "__" + self.task,
        )
        
        multi_vectors = MultiVectorEmbedding(
            id=id,
            embeddings=mv_data_resp['data'][0]["embeddings"],  # (755, 128)
            text=text,
            model_embed=self.model + "__" + self.task,
        )
        
        return EmbeddingResult(
            id=id,
            single_vector=single_vector,
            multi_vectors=multi_vectors
        )


    async def _make_api_call(self, payload: Dict[str, Any], return_multivector: bool) -> httpx.Response:
        """Make a single API call with rate limiting."""

        await self._rate_limit()  # Rate limit before each API call
        client = await self._get_client()
        response = await client.post(
            self.api_url,
            json={**payload, "return_multivector": return_multivector},
        )
        response.raise_for_status()
        return response


    async def _embed_single_item(self, embedable: Dict[str, Any]) -> EmbeddingResult:
        """Generate embeddings for a single embedable."""

        id = embedable.get("id")
        if not id:
            raise EmbeddingError("embedable must have 'id' field")
        
        input_payload, text = self._build_input_payload(embedable, id)

        payload = {
            "model": self.model,
            "task": self.task,
            "late_chunking": False,
            "truncate": True,
            "input": input_payload,
        }
        
        sv_response, mv_response = await asyncio.gather(
            self._make_api_call(payload, return_multivector=False),
            self._make_api_call(payload, return_multivector=True),
        )

        embedding_result = self._build_embedding_result(
            id, text, sv_response.json(), mv_response.json()
        )
        
        return embedding_result


    @retry_with_backoff(max_retries=3, base_delay=1.0)
    async def embed(self, embedables: List[Dict[str, Any]]) -> List[EmbeddingResult]:
        """Generate embeddings for a list of embedables."""
        
        if not self.api_key:
            raise EmbeddingError("Jina API key not configured")
        
        if not embedables:
            raise EmbeddingError("embedables list must not be empty")

        try:
            # Process items sequentially (rate limiter handles concurrency)
            # Each item's two API calls are parallelized internally
            embedding_results = []
            for embedable in embedables:
                result = await self._embed_single_item(embedable)
                embedding_results.append(result)

            return embedding_results

        except httpx.HTTPStatusError as e:
            logger.error(f"Jina API error: {e.response.status_code} - {e.response.text}")
            raise EmbeddingError(f"Jina API error: {e.response.status_code}")
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise EmbeddingError(f"Failed to generate embeddings: {e}")
    

    async def close(self):
        """Close HTTP client"""
        if self._client:
            await self._client.aclose()
            self._client = None
    
