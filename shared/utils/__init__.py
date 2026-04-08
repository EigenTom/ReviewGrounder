"""
Shared utilities for the unified review system.
"""
# Core utilities (always available)
from .json_parser import (
    parse_review_markdown,
    parse_keywords_json,
    parse_summary_json,
    parse_json_response,
)

from .prompt_loader import get_prompt_loader

# API Key Pool and Endpoint Pool (always available)
try:
    from .asta_api_key_pool import AstaAPIKeyPool
    _all_pools = ['AstaAPIKeyPool']
except ImportError:
    AstaAPIKeyPool = None
    _all_pools = []

try:
    from .vllm_endpoint_pool import VLLMEndpointPool
    _all_pools.append('VLLMEndpointPool')
except ImportError:
    VLLMEndpointPool = None

if _all_pools:
    __all__ = _all_pools

# Lazy imports for heavy dependencies (LLM-related)
# These may fail if dependencies are not installed, but that's okay
def _lazy_import_llm_services():
    """Lazy import LLM services to avoid dependency issues"""
    try:
        from .llm_service import LLMService, ChatMessage
        from .llm_service_factory import (
            get_llm_service_factory,
            LLMServiceFactory,
            load_api_key_from_config,
        )
        return {
            'LLMService': LLMService,
            'ChatMessage': ChatMessage,
            'get_llm_service_factory': get_llm_service_factory,
            'LLMServiceFactory': LLMServiceFactory,
            'load_api_key_from_config': load_api_key_from_config,
        }
    except ImportError:
        return {}

def _lazy_import_llm_implementations():
    """Lazy import LLM service implementations"""
    result = {}
    try:
        from .vllm_service import VLLMService
        result['VLLMService'] = VLLMService
    except ImportError:
        pass
    
    try:
        from .gpt_service import GPTService
        result['GPTService'] = GPTService
    except ImportError:
        pass
    
    try:
        from .mock_llm_service import MockLLMService, extract_title_from_latex, extract_abstract_from_latex
        result['MockLLMService'] = MockLLMService
        result['extract_title_from_latex'] = extract_title_from_latex
        result['extract_abstract_from_latex'] = extract_abstract_from_latex
    except ImportError:
        pass
    
    return result

def _lazy_import_other():
    """Lazy import other utilities"""
    result = {}
    try:
        from .reranker import rerank_paragraphs_bge
        result['rerank_paragraphs_bge'] = rerank_paragraphs_bge
    except ImportError:
        pass
    
    try:
        from .review_logger import ReviewLogger
        result['ReviewLogger'] = ReviewLogger
    except ImportError:
        pass
    
    return result

# Populate __all__ dynamically
_llm_services = _lazy_import_llm_services()
_llm_impls = _lazy_import_llm_implementations()
_other = _lazy_import_other()

# Make all lazy imports available at module level
globals().update(_llm_services)
globals().update(_llm_impls)
globals().update(_other)

__all__ = [
    'parse_review_markdown',
    'parse_keywords_json',
    'parse_summary_json',
    'parse_json_response',
    'get_prompt_loader',
] + list(_llm_services.keys()) + list(_llm_impls.keys()) + list(_other.keys())

if AstaAPIKeyPool:
    __all__.append('AstaAPIKeyPool')
