"""
Main pipeline for paper review
This module provides a high-level interface for the complete review pipeline
"""
import sys
import os
import yaml
import json
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Optional, Any, Tuple

# Suppress httpx INFO logs (HTTP Request messages)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Add project root to path for shared utils import
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from shared.utils.llm_service_factory import get_llm_service_factory
from shared.utils.llm_service import LLMService
from shared.utils.review_logger import ReviewLogger
from .paper_search import PaperRetriever
from .related_work_searcher import RelatedWorkSearcher
from .paper_reviewer import PaperReviewer
from .paper_results_analyzer import PaperResultsAnalyzer
from .paper_insight_miner import PaperInsightMiner
from .review_refiner import ReviewRefiner

try:
    from FlagEmbedding import FlagReranker
except ImportError:
    FlagReranker = None


def _load_config_file(config_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from config.yaml file
    
    Args:
        config_file: Optional path to config file. If None, uses default location.
        
    Returns:
        Configuration dictionary
    """
    if config_file is None:
        config_file = project_root / "shared" / "configs" / "config.yaml"
    
    config_path = Path(config_file)
    if not config_path.exists():
        # Return empty dict if config file doesn't exist
        return {}
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def create_reviewer_pipeline(
    vllm_base_url: Optional[str] = None,
    asta_api_key: Optional[str] = None,
    reranker_model: Optional[str] = None,
    reranker: Optional[FlagReranker] = None,
    max_related_papers: Optional[int] = None,
    prompts_file: Optional[str] = None,
    llm_config_file: Optional[str] = None,
    config_file: Optional[str] = None,
    keyword_llm_service: Optional[LLMService] = None,
    summarizer_llm_service: Optional[LLMService] = None,
    reviewer_llm_service: Optional[LLMService] = None,
    results_summarizer_llm_service: Optional[LLMService] = None,
    enable_logging: bool = False,
    log_dir: Optional[str] = None,
    logger: Optional[ReviewLogger] = None,
) -> PaperReviewer:
    """
    Create a complete reviewer pipeline with all components initialized
    
    Args:
        vllm_base_url: vLLM server URL (overrides config, deprecated - use llm_config_file)
        asta_api_key: Asta API key (overrides config, or set ASTA_API_KEY env var)
        reranker_model: Reranker model path (overrides config, ignored if reranker is provided)
        reranker: Optional pre-initialized reranker instance (if provided, will be reused instead of creating new one)
        max_related_papers: Maximum number of related papers to retrieve (overrides config)
        prompts_file: Optional path to prompts YAML file (default: configs/prompts.yaml)
        llm_config_file: Optional path to LLM service config YAML file (default: configs/llm_service_config.yaml)
        config_file: Optional path to main config YAML file (default: configs/config.yaml)
        keyword_llm_service: Optional LLM service for keyword generation (if None, uses config)
        summarizer_llm_service: Optional LLM service for paper summarization (if None, uses config)
        reviewer_llm_service: Optional LLM service for review generation (if None, uses config)
        results_summarizer_llm_service: Optional LLM service for paper results summarization (if None, uses paper_summarizer config)
        enable_logging: Whether to enable logging of intermediate outputs
        log_dir: Directory to save log files (only used if enable_logging=True and logger=None)
        logger: Optional pre-initialized ReviewLogger (if None and enable_logging=True, creates one)
        
    Returns:
        PaperReviewer instance
    """
    """
    Create a complete reviewer pipeline with all components initialized
    
    Args:
        vllm_base_url: vLLM server URL (overrides config, deprecated - use llm_config_file)
        asta_api_key: Asta API key (overrides config, or set ASTA_API_KEY env var)
        reranker_model: Reranker model path (overrides config, ignored if reranker is provided)
        reranker: Optional pre-initialized reranker instance (if provided, will be reused instead of creating new one)
        max_related_papers: Maximum number of related papers to retrieve (overrides config)
        prompts_file: Optional path to prompts YAML file (default: configs/prompts.yaml)
        llm_config_file: Optional path to LLM service config YAML file (default: configs/llm_service_config.yaml)
        config_file: Optional path to main config YAML file (default: configs/config.yaml)
        keyword_llm_service: Optional LLM service for keyword generation (if None, uses config)
        summarizer_llm_service: Optional LLM service for paper summarization (if None, uses config)
        reviewer_llm_service: Optional LLM service for review generation (if None, uses config)
        results_summarizer_llm_service: Optional LLM service for paper results summarization (if None, uses paper_summarizer config)
        enable_logging: Whether to enable logging of intermediate outputs
        log_dir: Directory to save log files (only used if enable_logging=True and logger=None)
        logger: Optional pre-initialized ReviewLogger (if None and enable_logging=True, creates one)
        
    Returns:
        Initialized PaperReviewer instance
    """
    # Load main config file
    config = _load_config_file(config_file)
    
    # Set default paths for prompts and LLM config if not provided
    if prompts_file is None:
        prompts_file = str(project_root / "shared" / "configs" / "prompts.yaml")
    if llm_config_file is None:
        llm_config_file = str(project_root / "shared" / "configs" / "llm_service_config.yaml")
    
    # Get LLM service factory
    factory = get_llm_service_factory(llm_config_file)
    
    # Create LLM services based on config or provided services
    if keyword_llm_service is None:
        keyword_component = factory.get_llm_assignment("keyword_generator")
        keyword_llm_service = factory.create_service_for_component("keyword_generator")
        if vllm_base_url and keyword_component != "gpt":
            # Override base_url if provided (for backward compatibility)
            keyword_llm_service.base_url = vllm_base_url
    
    if summarizer_llm_service is None:
        summarizer_component = factory.get_llm_assignment("paper_summarizer")
        summarizer_llm_service = factory.create_service_for_component("paper_summarizer")
        if vllm_base_url and summarizer_component != "gpt":
            summarizer_llm_service.base_url = vllm_base_url
    
    if reviewer_llm_service is None:
        reviewer_component = factory.get_llm_assignment("reviewer")
        reviewer_llm_service = factory.create_service_for_component("reviewer")
        if vllm_base_url and reviewer_component != "gpt":
            reviewer_llm_service.base_url = vllm_base_url
    
    # Create results analyzer LLM service
    if results_summarizer_llm_service is None:
        results_analyzer_component = factory.get_llm_assignment("results_analyzer")
        results_summarizer_llm_service = factory.create_service_for_component("results_analyzer")
        if vllm_base_url and results_analyzer_component != "gpt":
            results_summarizer_llm_service.base_url = vllm_base_url

    # Get paper search configs from config.yaml
    paper_search_config = config.get("paper_search", {})
    asta_config = paper_search_config.get("asta", {})
    reranker_config = paper_search_config.get("reranker", {})
    retrieval_config = paper_search_config.get("retrieval", {})
    
    # Use provided values or fall back to config.yaml, then env vars
    # Priority: parameter > api_key_pool_path (config) > api_key (config) > env var > None
    final_asta_api_key = asta_api_key or asta_config.get("api_key") or os.environ.get("ASTA_API_KEY")
    final_asta_api_key_pool_path = asta_config.get("api_key_pool_path")  # Prefer pool
    final_asta_endpoint = asta_config.get("endpoint")
    final_reranker_model = reranker_model or reranker_config.get("model")
    final_top_n = max_related_papers or retrieval_config.get("top_n", 10)
    final_use_abstract = retrieval_config.get("use_abstract", True)
    final_norm_cite = retrieval_config.get("norm_cite", False)
    final_min_citation = retrieval_config.get("min_citation")
    final_limit_per_keyword = retrieval_config.get("limit_per_keyword", 20)
    
    # Initialize paper retriever
    # Note: To avoid pickle issues, always pass the path even if reranker instance is provided
    # This allows lazy loading when needed, avoiding pickle issues
    paper_retriever = PaperRetriever.create_with_asta(
        api_key=final_asta_api_key if not final_asta_api_key_pool_path else None,  # If using pool, don't use single key
        api_key_pool_path=final_asta_api_key_pool_path,
        endpoint=final_asta_endpoint,
        reranker_model=final_reranker_model,  # Always pass path for lazy loading
        reranker=reranker,  # If provided, use with priority; otherwise will lazy load from path
        top_n=final_top_n,
        use_abstract=final_use_abstract,
        norm_cite=final_norm_cite,
        min_citation=final_min_citation,
    )
    
    # Get related work searcher configs from config.yaml
    related_work_config = config.get("related_work_searcher", {})
    final_max_related_papers = max_related_papers or related_work_config.get("max_related_papers", 10)
    final_max_parallel_summaries = related_work_config.get("max_parallel_summaries", 8)  # Default: 8 parallel workers
    
    # Initialize logger if enabled
    if logger is None and enable_logging:
        logger = ReviewLogger(log_dir=log_dir, enabled=True)
    elif not enable_logging:
        logger = None
    
    # Get verbose setting from config
    related_work_verbose = related_work_config.get("verbose", True)
    
    # Initialize related work searcher
    related_work_searcher = RelatedWorkSearcher(
        paper_retriever=paper_retriever,
        max_related_papers=final_max_related_papers,
        max_parallel_summaries=final_max_parallel_summaries,
        prompts_file=prompts_file,
        keyword_llm_service=keyword_llm_service,
        summarizer_llm_service=summarizer_llm_service,
        logger=logger,
        verbose=related_work_verbose,
    )
    
    # Initialize paper results analyzer
    paper_results_analyzer = PaperResultsAnalyzer(
        prompts_file=prompts_file,
        llm_service=results_summarizer_llm_service,
        logger=logger,
    )
    
    # Get paper reviewer config from config.yaml
    paper_reviewer_config = config.get("paper_reviewer", {})
    
    # Initialize paper reviewer
    paper_reviewer = PaperReviewer(
        reviewer_llm_service=reviewer_llm_service,
        prompts_file=prompts_file,
        logger=logger,
    )
    
    # Store reviewer config in reviewer instance for later use
    paper_reviewer.config = paper_reviewer_config
    
    return paper_reviewer


def review_paper_from_dict(
    paper_data: Dict[str, Any],
    reviewer: Optional[PaperReviewer] = None,
    **reviewer_kwargs
) -> Dict[str, Any]:
    """
    Review a paper from a dictionary
    
    Args:
        paper_data: Dictionary with 'title', 'abstract', optionally 'content' and 'keywords'
        reviewer: Optional pre-initialized reviewer (if None, creates one)
        **reviewer_kwargs: Arguments to pass to create_reviewer_pipeline if reviewer is None
        
    Returns:
        Review dictionary
    """
    if reviewer is None:
        reviewer = create_reviewer_pipeline(**reviewer_kwargs)
    
    return reviewer.review_paper(
        title=paper_data.get("title", ""),
        abstract=paper_data.get("abstract", ""),
        content=paper_data.get("content") or paper_data.get("text"),
        keywords=paper_data.get("keywords"),
        publication_date_range=paper_data.get("publication_date_range"),
        venues=paper_data.get("venues"),
        review_format=paper_data.get("review_format", "detailed"),
    )


def create_review_pipeline_with_refiner(
    vllm_base_url: Optional[str] = None,
    asta_api_key: Optional[str] = None,
    reranker_model: Optional[str] = None,
    reranker: Optional[FlagReranker] = None,
    force_direct_reranker: bool = False,
    max_related_papers: Optional[int] = None,
    prompts_file: Optional[str] = None,
    llm_config_file: Optional[str] = None,
    config_file: Optional[str] = None,
    keyword_llm_service: Optional[LLMService] = None,
    summarizer_llm_service: Optional[LLMService] = None,
    reviewer_llm_service: Optional[LLMService] = None,
    refiner_llm_service: Optional[LLMService] = None,
        results_summarizer_llm_service: Optional[LLMService] = None,
        insight_miner_llm_service: Optional[LLMService] = None,
        enable_logging: bool = False,
        log_dir: Optional[str] = None,
        logger: Optional[ReviewLogger] = None,
) -> tuple[PaperReviewer, Any, RelatedWorkSearcher, PaperResultsAnalyzer, PaperInsightMiner]:
    """
    Create a complete review pipeline with reviewer and refiner components
    
    This pipeline follows the new architecture:
    1. Reviewer: Generates initial review draft based only on paper content
    2. Refiner: Refines the draft using external tool information (related work, experiment results, method insights)
    
    Args:
        vllm_base_url: vLLM server URL (overrides config, deprecated - use llm_config_file)
        asta_api_key: Asta API key (overrides config, or set ASTA_API_KEY env var)
        reranker_model: Reranker model path (overrides config, ignored if reranker is provided)
        reranker: Optional pre-initialized reranker instance (if provided, will be reused instead of creating new one)
        max_related_papers: Maximum number of related papers to retrieve (overrides config)
        prompts_file: Optional path to prompts YAML file (default: configs/prompts.yaml)
        llm_config_file: Optional path to LLM service config YAML file (default: configs/llm_service_config.yaml)
        config_file: Optional path to main config YAML file (default: configs/config.yaml)
        keyword_llm_service: Optional LLM service for keyword generation (if None, uses config)
        summarizer_llm_service: Optional LLM service for paper summarization (if None, uses config)
        reviewer_llm_service: Optional LLM service for review generation (if None, uses config)
        refiner_llm_service: Optional LLM service for review refinement (if None, uses config)
        results_summarizer_llm_service: Optional LLM service for paper results summarization (if None, uses paper_summarizer config)
        insight_miner_llm_service: Optional LLM service for insight mining (if None, uses paper_summarizer config)
        enable_logging: Whether to enable logging of intermediate outputs
        log_dir: Directory to save log files (only used if enable_logging=True and logger=None)
        logger: Optional pre-initialized ReviewLogger (if None and enable_logging=True, creates one)
        
    Returns:
        Tuple of (reviewer, refiner, related_work_searcher, paper_results_analyzer, paper_insight_miner)
    """
    # Load main config file
    config = _load_config_file(config_file)
    
    # Set default paths for prompts and LLM config if not provided
    if prompts_file is None:
        prompts_file = str(project_root / "shared" / "configs" / "prompts.yaml")
    if llm_config_file is None:
        llm_config_file = str(project_root / "shared" / "configs" / "llm_service_config.yaml")
    
    # Get LLM service factory
    factory = get_llm_service_factory(llm_config_file)
    
    # Create LLM services based on config or provided services
    if keyword_llm_service is None:
        keyword_component = factory.get_llm_assignment("keyword_generator")
        keyword_llm_service = factory.create_service_for_component("keyword_generator")
        if vllm_base_url and keyword_component != "gpt":
            keyword_llm_service.base_url = vllm_base_url
    
    if summarizer_llm_service is None:
        summarizer_component = factory.get_llm_assignment("paper_summarizer")
        summarizer_llm_service = factory.create_service_for_component("paper_summarizer")
        if vllm_base_url and summarizer_component != "gpt":
            summarizer_llm_service.base_url = vllm_base_url
    
    if reviewer_llm_service is None:
        reviewer_component = factory.get_llm_assignment("reviewer")
        reviewer_llm_service = factory.create_service_for_component("reviewer")
        if vllm_base_url and reviewer_component != "gpt":
            reviewer_llm_service.base_url = vllm_base_url
    
    # Create refiner LLM service (defaults to reviewer if not provided)
    if refiner_llm_service is None:
        try:
            refiner_component = factory.get_llm_assignment("refiner")
            refiner_llm_service = factory.create_service_for_component("refiner")
            if vllm_base_url and refiner_component != "gpt":
                refiner_llm_service.base_url = vllm_base_url
        except (KeyError, AttributeError, ValueError):
            # Fallback to reviewer service if refiner not configured or service creation fails
            print("Warning: refiner not configured in llm_assignments, using reviewer service")
            refiner_llm_service = reviewer_llm_service
    
    # Create results analyzer LLM service
    if results_summarizer_llm_service is None:
        results_analyzer_component = factory.get_llm_assignment("results_analyzer")
        results_summarizer_llm_service = factory.create_service_for_component("results_analyzer")
        if vllm_base_url and results_analyzer_component != "gpt":
            results_summarizer_llm_service.base_url = vllm_base_url

    # Create insight miner LLM service
    if insight_miner_llm_service is None:
        insight_miner_component = factory.get_llm_assignment("insight_miner")
        insight_miner_llm_service = factory.create_service_for_component("insight_miner")
        if vllm_base_url and insight_miner_component != "gpt":
            insight_miner_llm_service.base_url = vllm_base_url
    
    # Get paper search configs from config.yaml
    paper_search_config = config.get("paper_search", {})
    asta_config = paper_search_config.get("asta", {})
    reranker_config = paper_search_config.get("reranker", {})
    reranker_api_config = reranker_config.get("api", {})
    retrieval_config = paper_search_config.get("retrieval", {})
    
    # Use provided values or fall back to config.yaml, then env vars
    # Priority: parameter > api_key_pool_path (config) > api_key (config) > env var > None
    # Prefer explicit parameter, then environment variable, then config file
    final_asta_api_key = asta_api_key or os.environ.get("ASTA_API_KEY") or asta_config.get("api_key")
    final_asta_api_key_pool_path = asta_config.get("api_key_pool_path")  # Prefer pool
    final_asta_endpoint = asta_config.get("endpoint")
    final_reranker_model = reranker_model or reranker_config.get("model")
    
    # Reranker API configuration
    final_reranker_api_base_url = reranker_api_config.get("base_url")
    final_reranker_api_endpoint_pool_path = reranker_api_config.get("endpoint_pool_path")
    final_reranker_api_timeout = reranker_api_config.get("timeout", 30.0)

    # In single-paper inference mode we may want to force local reranker usage
    # even if API endpoints are configured in the shared config.
    if force_direct_reranker:
        final_reranker_api_base_url = None
        final_reranker_api_endpoint_pool_path = None
    
    final_top_n = max_related_papers or retrieval_config.get("top_n", 10)
    final_use_abstract = retrieval_config.get("use_abstract", True)
    final_norm_cite = retrieval_config.get("norm_cite", False)
    final_min_citation = retrieval_config.get("min_citation")
    final_limit_per_keyword = retrieval_config.get("limit_per_keyword", 20)
    
    # Initialize paper retriever
    # Note: To avoid pickle issues, always pass the path even if reranker instance is provided
    # This allows lazy loading when needed, avoiding pickle issues
    # If reranker API is configured, prefer API mode
    paper_retriever = PaperRetriever.create_with_asta(
        api_key=final_asta_api_key if not final_asta_api_key_pool_path else None,  # If using pool, don't use single key
        api_key_pool_path=final_asta_api_key_pool_path,
        endpoint=final_asta_endpoint,
        reranker_model=final_reranker_model if not (final_reranker_api_base_url or final_reranker_api_endpoint_pool_path) else None,  # If using API mode, don't pass model path
        reranker=reranker if not (final_reranker_api_base_url or final_reranker_api_endpoint_pool_path) else None,  # If using API mode, don't use direct reranker
        reranker_api_base_url=final_reranker_api_base_url,
        reranker_api_endpoint_pool_path=final_reranker_api_endpoint_pool_path,
        reranker_api_timeout=final_reranker_api_timeout,
        top_n=final_top_n,
        use_abstract=final_use_abstract,
        norm_cite=final_norm_cite,
        min_citation=final_min_citation,
    )
    
    # Get related work searcher configs from config.yaml
    related_work_config = config.get("related_work_searcher", {})
    final_max_related_papers = max_related_papers or related_work_config.get("max_related_papers", 10)
    final_max_parallel_summaries = related_work_config.get("max_parallel_summaries", 8)  # Default: 8 parallel workers
    
    # Initialize logger if enabled
    if logger is None and enable_logging:
        logger = ReviewLogger(log_dir=log_dir, enabled=True)
    elif not enable_logging:
        logger = None
    
    # Get verbose setting from config
    related_work_verbose = related_work_config.get("verbose", True)
    
    # Initialize related work searcher (domain-specific tool)
    related_work_searcher = RelatedWorkSearcher(
        paper_retriever=paper_retriever,
        max_related_papers=final_max_related_papers,
        max_parallel_summaries=final_max_parallel_summaries,
        prompts_file=prompts_file,
        keyword_llm_service=keyword_llm_service,
        summarizer_llm_service=summarizer_llm_service,
        logger=logger,
        verbose=related_work_verbose,
    )
    
    # Initialize paper results summarizer (domain-specific tool)
    paper_results_analyzer = PaperResultsAnalyzer(
        prompts_file=prompts_file,
        llm_service=results_summarizer_llm_service,
        logger=logger,
    )
    
    # Initialize paper insight miner (domain-specific tool)
    paper_insight_miner = PaperInsightMiner(
        prompts_file=prompts_file,
        llm_service=insight_miner_llm_service,
        logger=logger,
    )
    
    # Get paper reviewer config from config.yaml
    paper_reviewer_config = config.get("paper_reviewer", {})
    
    # Initialize paper reviewer (no external tools - generates initial draft)
    paper_reviewer = PaperReviewer(
        reviewer_llm_service=reviewer_llm_service,
        prompts_file=prompts_file,
        logger=logger,
    )
    
    # Store reviewer config in reviewer instance for later use
    paper_reviewer.config = paper_reviewer_config
    
    # Get refiner config from config.yaml
    refiner_config = config.get("review_refiner", {})
    
    # Initialize review refiner (uses external tool information)
    review_refiner = ReviewRefiner(
        refiner_llm_service=refiner_llm_service,
        prompts_file=prompts_file,
        logger=logger,
    )
    
    # Store refiner config in refiner instance for later use
    review_refiner.config = refiner_config
    
    return paper_reviewer, review_refiner, related_work_searcher, paper_results_analyzer, paper_insight_miner


def review_paper_with_refiner(
    paper_data: Dict[str, Any],
    reviewer: Optional[PaperReviewer] = None,
    refiner: Optional[Any] = None,
    related_work_searcher: Optional[RelatedWorkSearcher] = None,
    paper_results_analyzer: Optional[PaperResultsAnalyzer] = None,
    paper_insight_miner: Optional[PaperInsightMiner] = None,
    verbose: Optional[bool] = None,  # Override global verbose setting
    **pipeline_kwargs
) -> Dict[str, Any]:
    """
    Review a paper using the new pipeline: reviewer -> refiner
    
    Args:
        paper_data: Dictionary with 'title', 'abstract', optionally 'content' and 'keywords'
        reviewer: Optional pre-initialized reviewer (if None, creates one)
        refiner: Optional pre-initialized refiner (if None, creates one)
        related_work_searcher: Optional pre-initialized related work searcher (if None, creates one)
        paper_results_analyzer: Optional pre-initialized paper results analyzer (if None, creates one)
        **pipeline_kwargs: Arguments to pass to create_review_pipeline_with_refiner if components are None
        
    Returns:
        Refined review dictionary
    """
    # Create components if not provided
    # if reviewer is None or refiner is None or related_work_searcher is None or paper_results_analyzer is None or paper_insight_miner is None:
    #     reviewer, refiner, related_work_searcher, paper_results_analyzer, paper_insight_miner = create_review_pipeline_with_refiner(**pipeline_kwargs)
    
    title = paper_data.get("title", "")
    abstract = paper_data.get("abstract", "")
    content = paper_data.get("content") or paper_data.get("text")
    
    keywords = paper_data.get("keywords")
    publication_date_range = paper_data.get("publication_date_range")
    venues = paper_data.get("venues")
    
    # Get review_format from paper_data or config
    # Load config if not already loaded
    if reviewer is None or refiner is None:
        config = _load_config_file(pipeline_kwargs.get("config_file"))
        paper_reviewer_config = config.get("paper_reviewer", {})
        review_format = paper_data.get("review_format") or paper_reviewer_config.get("review_format", "detailed")
    else:
        config = _load_config_file(pipeline_kwargs.get("config_file") if pipeline_kwargs else None)
        review_format = paper_data.get("review_format", "detailed")
    
    # Get global verbose setting from config (if not explicitly provided)
    if verbose is None:
        verbose = config.get("verbose", True)  # Default to True for backward compatibility
    
    # Step 1: Generate initial review draft (reviewer, no external tools)
    # Don't auto-save log here, we'll save after refiner completes
    initial_review = reviewer.review_paper(
        title=title,
        abstract=abstract,
        content=content,
        keywords=keywords,
        publication_date_range=publication_date_range,
        venues=venues,
        review_format=review_format,
        auto_save_log=False,  # Don't save log yet, wait for refiner
        verbose=verbose,  # Pass verbose to reviewer
    )
    
    # Extract initial scores from paper_data if available (for pre-generated reviews)
    initial_scores = paper_data.get('initial_scores', {})
    if initial_scores:
        # Add initial scores to initial_review so refiner can access them
        initial_review['initial_scores'] = initial_scores
    
    # Step 2: Get external tool information
    if verbose:
        print("\n" + "=" * 60)
        print("Step 2: Gathering external tool information")
        print("=" * 60)
    
    # Get paper results analyzer output
    results_analyzer_json = None
    if paper_results_analyzer and content:
        if verbose:
            print("Getting paper results analyzer output...")
        try:
            # Format initial review as text for the analyzer
            initial_review_text = refiner._format_review_dict(initial_review, review_format) if hasattr(refiner, '_format_review_dict') else str(initial_review)
            results_analyzer_json = paper_results_analyzer.analyze_paper_results(content, initial_review_text)
        except Exception as e:
            print(f"Error generating paper results analyzer output: {e}")
            traceback.print_exc()
            return None

    def _run_insight_miner() -> Optional[str]:
        if not paper_insight_miner or not content:
            return None
        try:
            return paper_insight_miner.mine_paper_insights(content, initial_review_text)
        except Exception as e:
            print(f"Error generating paper insight miner output: {e}")
            traceback.print_exc()
            return None

    def _run_related_work() -> Tuple[Optional[str], Optional[list]]:
        if not related_work_searcher:
            return None, None
        try:
            related_work_json_list = related_work_searcher.generate_related_work_json_list(
                title=title,
                abstract=abstract,
                content=content,
                keywords=keywords,
                publication_date_range=publication_date_range,
                venues=venues,
            )
            kw = getattr(related_work_searcher, "last_keywords", None)
            return json.dumps(related_work_json_list, ensure_ascii=False), kw
        except Exception as e:
            print(f"Error generating related work JSON list: {e}")
            traceback.print_exc()
            return None, None

    with ThreadPoolExecutor(max_workers=3) as executor:
        future_ra = executor.submit(_run_results_analyzer) if (paper_results_analyzer and content) else None
        future_im = executor.submit(_run_insight_miner) if (paper_insight_miner and content) else None
        future_rw = executor.submit(_run_related_work) if related_work_searcher else None

        if future_ra is not None:
            results_analyzer_json = future_ra.result()
        if future_im is not None:
            insight_miner_json = future_im.result()
        if future_rw is not None:
            related_work_json_list_str, search_keywords = future_rw.result()
    
    # Step 3: Refine review with external tool information
    # Get refiner-specific review_format from config if available
    # Try to get config from pipeline_kwargs or load it
    try:
        if config is None:
            config = _load_config_file(pipeline_kwargs.get("config_file") if pipeline_kwargs else None)
        refiner_config = config.get("review_refiner", {})
        refiner_review_format = paper_data.get("refiner_review_format") or refiner_config.get("review_format") or review_format
    except:
        refiner_review_format = review_format
    
    refined_review = refiner.refine_review(
        initial_review=initial_review,
        insight_miner_json=insight_miner_json,
        results_analyzer_json=results_analyzer_json,
        related_work_json_list=related_work_json_list_str,
        title=title,
        abstract=abstract,
        content=content,
        review_format=refiner_review_format,
        verbose=verbose,  # Pass verbose to refiner
    )
    
    # Attach search keywords and related-work helpers for downstream UIs.
    if search_keywords is not None:
        refined_review["search_keywords"] = search_keywords
    
    # Save log if logger is available
    if hasattr(reviewer, 'logger') and reviewer.logger:
        log_path = reviewer.logger.save_run()
        if log_path and verbose:
            print(f"\nSaved complete log to: {log_path}")
    
    return refined_review
