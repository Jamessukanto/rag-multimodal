"""
Similarity functions (maxsim, cosine, dot product)
"""

import numpy as np
from typing import List


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Compute cosine similarity between two vectors"""
    vec1_np = np.array(vec1)
    vec2_np = np.array(vec2)
    
    dot_product = np.dot(vec1_np, vec2_np)
    norm1 = np.linalg.norm(vec1_np)
    norm2 = np.linalg.norm(vec2_np)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(dot_product / (norm1 * norm2))


def maxsim_score(
    query_multi_vectors: List[List[float]],
    chunk_multi_vectors: List[List[float]]
) -> float:
    """
    Compute MaxSim score between query and chunk multi-vectors (vectorized).
    
    MaxSim = sum(max(cosine(query_vec_i, chunk_vec_j) for all chunk_vec_j) for all query_vec_i)
    
    For each query token vector, find the maximum similarity with any chunk token vector,
    then sum these maximum similarities across all query tokens.
    
    Uses vectorized numpy operations for performance.
    
    Args:
        query_multi_vectors: List of query token embedding vectors
        chunk_multi_vectors: List of chunk token embedding vectors
    
    Returns:
        MaxSim score
    """
    if not query_multi_vectors or not chunk_multi_vectors:
        return 0.0
    
    # Convert to numpy arrays for vectorized operations
    query_arr = np.array(query_multi_vectors)  # Shape: [Nq, d]
    chunk_arr = np.array(chunk_multi_vectors)  # Shape: [Nc, d]
    
    # Normalize vectors (for cosine similarity)
    query_norm = np.linalg.norm(query_arr, axis=1, keepdims=True)
    chunk_norm = np.linalg.norm(chunk_arr, axis=1, keepdims=True)
    
    # Avoid division by zero
    query_arr = query_arr / (query_norm + 1e-8)
    chunk_arr = chunk_arr / (chunk_norm + 1e-8)
    
    # Compute similarity matrix: [Nq, Nc]
    # Each element (i, j) is cosine similarity between query_vec_i and chunk_vec_j
    similarity_matrix = np.matmul(query_arr, chunk_arr.T)
    
    # For each query token, find max similarity with any chunk token: [Nq]
    max_similarities = np.max(similarity_matrix, axis=1)
    
    # Sum all max similarities
    total_maxsim = float(np.sum(max_similarities))
    
    return total_maxsim


def dot_product(vec1: List[float], vec2: List[float]) -> float:
    """Compute dot product between two vectors"""
    return float(np.dot(np.array(vec1), np.array(vec2)))

