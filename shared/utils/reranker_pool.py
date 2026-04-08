"""
Reranker Pool for multiprocessing-safe reranker sharing

Solve the issue of FlagReranker not being pickleable in multi-process/multi-thread environment.
"""
import os
import threading
from typing import Optional, Dict
from pathlib import Path

# Global reranker storage (thread-safe)
# Note: Dictionary access is atomic in Python for simple operations,
# but we use a lock for thread-safety when modifying the dict
_reranker_pool: Dict[str, object] = {}
_reranker_lock = threading.Lock()


def get_reranker(model_path: str, use_fp16: bool = True):
    """
    Get or create reranker (thread-safe, process-shared)
    
    Performance optimization:
    - Load and cache on first call
    - Return cached instance on subsequent calls (no lock check)
    - Use double-check locking pattern, reduce lock contention
    
    Args:
        model_path: Reranker model path
        use_fp16: whether to use FP16
        
    Returns:
        FlagReranker instance
    """
    global _reranker_pool
    
    # create unique key
    key = f"{model_path}_{use_fp16}"
    
    # performance optimization: fast path check (no lock)
    if key in _reranker_pool:
        return _reranker_pool[key]
    
    # slow path: needs loading (needs lock)
    with _reranker_lock:
        # double check: other threads may have loaded while waiting for lock
        if key not in _reranker_pool:
            # lazy import, avoid importing when module is loaded
            try:
                from FlagEmbedding import FlagReranker
                
                # set environment variable to suppress progress bar
                original_verbosity = os.environ.get('TRANSFORMERS_VERBOSITY', '')
                os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
                
                try:
                    # load model
                    reranker = FlagReranker(model_path, use_fp16=use_fp16)
                    _reranker_pool[key] = reranker
                finally:
                    # restore original verbosity
                    if original_verbosity:
                        os.environ['TRANSFORMERS_VERBOSITY'] = original_verbosity
                    elif 'TRANSFORMERS_VERBOSITY' in os.environ:
                        del os.environ['TRANSFORMERS_VERBOSITY']
                
            except ImportError:
                raise ImportError(
                    "FlagEmbedding not installed. Install it with: pip install FlagEmbedding"
                )
        
        return _reranker_pool[key]


def clear_reranker_pool():
    """clear reranker pool (mainly for testing)"""
    global _reranker_pool
    with _reranker_lock:
        _reranker_pool.clear()
