"""
Reranker utilities for paper retrieval
Based on OpenScholar's rerank_paragraphs_bge function

Supports two modes:
1. Direct mode: Use FlagReranker directly (requires global lock for thread-safety)
2. API mode: Use reranker API service with load balancing (recommended for multi-GPU)
"""
import os
import threading
import time
import requests
from typing import List, Dict, Any, Optional, Tuple, Union

# Suppress transformers progress bars
os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')

# Global lock for reranker usage (FlagReranker's tokenizer is not thread-safe)
# This prevents "Already borrowed" errors when multiple threads use the same reranker
# NOTE: Not needed when using API mode
_reranker_usage_lock = threading.Lock()

# Try to import endpoint pool for API mode
try:
    from .reranker_endpoint_pool import RerankerEndpointPool
    HAS_ENDPOINT_POOL = True
except ImportError:
    HAS_ENDPOINT_POOL = False
    RerankerEndpointPool = None


def rerank_paragraphs_bge(
    query: str,
    paragraphs: List[Dict[str, Any]],
    reranker: Optional[Any] = None,
    reranker_endpoint_pool: Optional[Any] = None,
    norm_cite: bool = False,
    start_index: int = 0,
    use_abstract: bool = False,
    timeout: float = 30.0,
) -> Tuple[List[Dict[str, Any]], Dict[int, float], Dict[int, int]]:
    """
    Rerank paragraphs using BGE reranker (from OpenScholar)
    
    Supports two modes:
    1. Direct mode: Pass FlagReranker instance (uses global lock, thread-safe but serialized)
    2. API mode: Pass RerankerEndpointPool (recommended for multi-GPU, parallel requests)
    
    Args:
        query: Search query
        paragraphs: List of paragraph/paper dictionaries
        reranker: FlagReranker instance (for direct mode, optional if using API mode)
        reranker_endpoint_pool: RerankerEndpointPool instance (for API mode, optional if using direct mode)
        norm_cite: Whether to normalize citation counts and add to scores
        start_index: Starting index for id mapping
        use_abstract: Whether to include abstract in reranking text
        timeout: Request timeout for API mode (seconds)
        
    Returns:
        Tuple of:
        - reranked_paragraphs: List of reranked paragraphs
        - result_dict: Dictionary mapping original index to score
        - id_mapping: Dictionary mapping new index to original index
    """
    # Filter out paragraphs without text
    paragraphs = [p for p in paragraphs if p.get("text") is not None]
    
    if not paragraphs:
        return [], {}, {}
    
    # Build paragraph texts for reranking
    if use_abstract:
        paragraph_texts = [
            p["title"] + "\n" + p["abstract"] + "\n" + p["text"]
            if "title" in p and "abstract" in p and p.get("title") and p.get("abstract")
            else p["text"]
            for p in paragraphs
        ]
    else:
        paragraph_texts = [
            p["title"] + " " + p["text"]
            if "title" in p and p.get("title") is not None
            else p["text"]
            for p in paragraphs
        ]
    
    # Filter out empty or None texts
    valid_indices = []
    valid_texts = []
    for i, text in enumerate(paragraph_texts):
        if text and isinstance(text, str) and text.strip():
            valid_indices.append(i)
            valid_texts.append(text)
    
    # If no valid texts, return empty results
    if not valid_texts:
        return [], {}, {}
    
    # If some texts were filtered out, update paragraphs list
    if len(valid_indices) < len(paragraphs):
        paragraphs = [paragraphs[i] for i in valid_indices]
        paragraph_texts = valid_texts
    
    # Compute reranking scores
    if reranker is None and reranker_endpoint_pool is None:
        # If no reranker, return original order
        id_mapping = {i: i + start_index for i in range(len(paragraphs))}
        result_dict = {i: 0.0 for i in range(len(paragraphs))}
        return paragraphs, result_dict, id_mapping
    
    # API mode: Use reranker API service (recommended for multi-GPU)
    if reranker_endpoint_pool is not None:
        return _rerank_via_api(
            query=query,
            paragraph_texts=paragraph_texts,
            paragraphs=paragraphs,
            reranker_endpoint_pool=reranker_endpoint_pool,
            norm_cite=norm_cite,
            start_index=start_index,
            timeout=timeout
        )
    
    # Direct mode: Use FlagReranker directly (requires global lock)
    # Suppress transformers warnings and progress bars during computation
    original_verbosity = os.environ.get('TRANSFORMERS_VERBOSITY', '')
    os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
    
    # Use lock to prevent "Already borrowed" errors from Rust tokenizer
    # FlagReranker's tokenizer is not thread-safe, so we need to serialize access
    with _reranker_usage_lock:
        try:
            # Ensure we have at least one valid text before calling compute_score
            if not paragraph_texts:
                return [], {}, {}
            scores = reranker.compute_score([[query, p] for p in paragraph_texts], batch_size=100)
        finally:
            # Restore original verbosity
            if original_verbosity:
                os.environ['TRANSFORMERS_VERBOSITY'] = original_verbosity
            elif 'TRANSFORMERS_VERBOSITY' in os.environ:
                del os.environ['TRANSFORMERS_VERBOSITY']
    
    # Handle score format (can be float or list)
    if isinstance(scores, float):
        result_dict = {0: scores}
    else:
        result_dict = {p_id: score for p_id, score in enumerate(scores)}
    
    # Add normalized citation counts if enabled
    if norm_cite:
        citation_items = [
            item["citation_counts"]
            for item in paragraphs
            if "citation_counts" in item and item["citation_counts"] is not None
        ]
        if len(citation_items) > 0:
            max_citations = max(citation_items)
            for p_id in result_dict:
                if (
                    "citation_counts" in paragraphs[p_id]
                    and paragraphs[p_id]["citation_counts"] is not None
                ):
                    result_dict[p_id] = result_dict[p_id] + (
                        paragraphs[p_id]["citation_counts"] / max_citations
                    )
    
    # Sort by score
    p_ids = sorted(result_dict.items(), key=lambda x: x[1], reverse=True)
    
    # Build reranked list and id mapping
    new_orders = []
    id_mapping = {}
    for i, (p_id, _) in enumerate(p_ids):
        new_orders.append(paragraphs[p_id])
        id_mapping[i] = int(p_id) + start_index
    
    return new_orders, result_dict, id_mapping


def _rerank_via_api(
    query: str,
    paragraph_texts: List[str],
    paragraphs: List[Dict[str, Any]],
    reranker_endpoint_pool: Any,
    norm_cite: bool = False,
    start_index: int = 0,
    timeout: float = 30.0,
) -> Tuple[List[Dict[str, Any]], Dict[int, float], Dict[int, int]]:
    """
    Rerank paragraphs via API service (supports load balancing across multiple GPUs)
    
    Args:
        query: Search query
        paragraph_texts: List of paragraph texts (already formatted)
        paragraphs: List of paragraph dictionaries
        reranker_endpoint_pool: RerankerEndpointPool instance
        norm_cite: Whether to normalize citation counts
        start_index: Starting index for id mapping
        timeout: Request timeout
        
    Returns:
        Tuple of reranked paragraphs, result dict, and id mapping
    """
    if not paragraph_texts:
        return [], {}, {}
    
    # Get endpoint from pool (round-robin load balancing)
    endpoint = reranker_endpoint_pool.get_endpoint()
    api_url = f"{endpoint}/rerank"
    
    # Prepare request
    request_data = {
        "query": query,
        "paragraphs": paragraph_texts,
        "batch_size": 100
    }
    
    start_time = time.time()
    try:
        # Make API request
        response = requests.post(
            api_url,
            json=request_data,
            timeout=timeout
        )
        response.raise_for_status()
        
        result = response.json()
        scores = result.get("scores", [])
        response_time = time.time() - start_time
        
        # Mark success
        reranker_endpoint_pool.mark_success(endpoint, response_time)
        
    except requests.exceptions.RequestException as e:
        # Mark error
        reranker_endpoint_pool.mark_error(endpoint, str(e))
        raise RuntimeError(f"Reranker API request failed: {e}")
    
    # Handle score format (should be list from API)
    if isinstance(scores, float):
        result_dict = {0: scores}
    else:
        result_dict = {p_id: score for p_id, score in enumerate(scores)}
    
    # Add normalized citation counts if enabled
    if norm_cite:
        citation_items = [
            item["citation_counts"]
            for item in paragraphs
            if "citation_counts" in item and item["citation_counts"] is not None
        ]
        if len(citation_items) > 0:
            max_citations = max(citation_items)
            for p_id in result_dict:
                if (
                    "citation_counts" in paragraphs[p_id]
                    and paragraphs[p_id]["citation_counts"] is not None
                ):
                    result_dict[p_id] = result_dict[p_id] + (
                        paragraphs[p_id]["citation_counts"] / max_citations
                    )
    
    # Sort by score
    p_ids = sorted(result_dict.items(), key=lambda x: x[1], reverse=True)
    
    # Build reranked list and id mapping
    new_orders = []
    id_mapping = {}
    for i, (p_id, _) in enumerate(p_ids):
        new_orders.append(paragraphs[p_id])
        id_mapping[i] = int(p_id) + start_index
    
    return new_orders, result_dict, id_mapping

