"""
Evaluation metrics: Recall@k, MRR, nDCG@k
"""

from typing import List, Set


def recall_at_k(relevant: Set[str], retrieved: List[str], k: int) -> float:
    """
    Compute Recall@k.
    
    Args:
        relevant: Set of relevant document IDs
        retrieved: List of retrieved document IDs (ordered)
        k: Cutoff value
        
    Returns:
        Recall@k score (0.0 to 1.0)
    """
    if not relevant:
        return 0.0
    
    retrieved_k = set(retrieved[:k])
    relevant_retrieved = len(relevant & retrieved_k)
    
    return relevant_retrieved / len(relevant)


def mrr(relevant: Set[str], retrieved: List[str]) -> float:
    """
    Compute Mean Reciprocal Rank (MRR).
    
    Args:
        relevant: Set of relevant document IDs
        retrieved: List of retrieved document IDs (ordered)
        
    Returns:
        MRR score (0.0 to 1.0)
    """
    if not relevant:
        return 0.0
    
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    
    return 0.0


def ndcg_at_k(relevant: Set[str], retrieved: List[str], k: int) -> float:
    """
    Compute nDCG@k (normalized Discounted Cumulative Gain).
    
    Args:
        relevant: Set of relevant document IDs
        retrieved: List of retrieved document IDs (ordered)
        k: Cutoff value
        
    Returns:
        nDCG@k score (0.0 to 1.0)
    """
    if not relevant:
        return 0.0
    
    # DCG@k
    dcg = 0.0
    for i, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant:
            dcg += 1.0 / (i * (i + 1) / 2)  # Simplified: relevance = 1 if relevant
    
    # IDCG@k (ideal DCG)
    idcg = 0.0
    num_relevant = min(len(relevant), k)
    for i in range(1, num_relevant + 1):
        idcg += 1.0 / (i * (i + 1) / 2)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg

