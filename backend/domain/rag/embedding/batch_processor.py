"""
Batch processing utilities for embeddings
"""

import logging
import asyncio
from typing import List, Dict, Any, Callable, Optional
from tqdm.asyncio import tqdm
from domain.rag.embedding.types import EmbeddingResult

logger = logging.getLogger(__name__)


class BatchProcessor:
    """Batch processing with progress tracking"""
    
    def __init__(self, batch_size: int = 10, max_concurrent: int = 5):
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
    
    async def process_batch(
        self,
        items: List[Dict[str, Any]],
        process_fn: Callable,
        show_progress: bool = True
    ) -> List[Any]:
        """
        Process items in batches with concurrency control.
        
        Args:
            items: List of items to process
            process_fn: Async function to process each item
            show_progress: Whether to show progress bar
            
        Returns:
            List of results
        """
        semaphore = asyncio.Semaphore(self.max_concurrent)
        results = []
        
        async def process_with_semaphore(item):
            async with semaphore:
                return await process_fn(item)
        
        tasks = [process_with_semaphore(item) for item in items]
        
        if show_progress:
            results = []
            for coro in tqdm.as_completed(tasks, total=len(tasks)):
                result = await coro
                results.append(result)
        else:
            results = await asyncio.gather(*tasks)
        
        return results
    
    async def process_in_batches(
        self,
        items: List[Dict[str, Any]],
        process_batch_fn: Callable,
        show_progress: bool = True
    ) -> List[Any]:
        """
        Process items in fixed-size batches.
        
        Args:
            items: List of items to process
            process_batch_fn: Async function to process a batch
            show_progress: Whether to show progress bar
            
        Returns:
            List of results
        """
        all_results = []
        
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            batch_results = await process_batch_fn(batch)
            all_results.extend(batch_results)
            
            if show_progress:
                logger.info(f"Processed batch {i // self.batch_size + 1}/{(len(items) + self.batch_size - 1) // self.batch_size}")
        
        return all_results

