"""
Unified paper retrieval and reranking class
Integrates search API and reranking functionality
"""
from typing import List, Dict, Optional, Any
from .paper_search_api import PaperSearchAPI

from shared.utils.reranker import rerank_paragraphs_bge

try:
    from FlagEmbedding import FlagReranker
except ImportError:
    FlagReranker = None


class PaperRetriever:
    """
    Unified paper retrieval and reranking class
    
    Features:
    1. Accepts query
    2. Calls search API to retrieve papers
    3. Uses reranker for reranking
    4. Returns TopN results
    """
    
    def __init__(self, 
                 search_api: PaperSearchAPI,
                 reranker: Optional[FlagReranker] = None,
                 reranker_model_path: Optional[str] = None,
                 reranker_use_fp16: bool = True,
                 reranker_api_base_url: Optional[str] = None,
                 reranker_api_endpoint_pool_path: Optional[str] = None,
                 reranker_api_timeout: float = 30.0,
                 top_n: int = 10,
                 use_abstract: bool = False,
                 norm_cite: bool = False,
                 min_citation: Optional[int] = None):
        """
        Initialize PaperRetriever
        
        Args:
            search_api: Paper search API instance (SemanticScholarAPI or AstaAPI)
            reranker: BGE reranker model instance (if None, will try to load from reranker_model_path)
            reranker_model_path: Reranker model path (only used when reranker is None, for lazy loading)
            reranker_use_fp16: Whether to use FP16 (only used when loading from path)
            top_n: Number of TopN papers to return
            use_abstract: Whether to use abstract in reranking
            norm_cite: Whether to normalize citation counts and add to reranking score
            min_citation: Minimum citation count filter
        """
        self.search_api = search_api
        self.reranker = reranker
        self._reranker_model_path = reranker_model_path  # Save path for lazy loading
        self._reranker_use_fp16 = reranker_use_fp16
        
        # Reranker API configuration (for API mode)
        self._reranker_api_base_url = reranker_api_base_url
        self._reranker_api_endpoint_pool_path = reranker_api_endpoint_pool_path
        self._reranker_api_timeout = reranker_api_timeout
        self._reranker_endpoint_pool = None  # Will be initialized on first use
        
        self.top_n = top_n
        self.use_abstract = use_abstract
        self.norm_cite = norm_cite
        self.min_citation = min_citation
    
    def retrieve(self, query: str, limit: int = 50, 
                publication_date_range: Optional[str] = None,
                venues: Optional[str] = None,
                fields: Optional[str] = None,
                mode: Optional[str] = None,
                skip_reranking: bool = False,
                **search_kwargs) -> List[Dict[str, Any]]:
        """
        Retrieve and rerank papers
        
        Args:
            query: Search query
            limit: Number of papers returned by initial search (before reranking)
            publication_date_range: Publication date range (Asta API supports)
            venues: Venue restrictions (Asta API supports)
            fields: Fields to return (Asta API supports, default "title,abstract,year")
            mode: Search mode ("query" or "title")
            skip_reranking: If True, skip reranking and return search results directly
            **search_kwargs: Additional parameters passed to search API
        
        Returns:
            Reranked TopN paper list (if skip_reranking=True, returns all search results)
        """
        # Build search parameters
        search_params = {}
        if publication_date_range is not None:
            search_params["publication_date_range"] = publication_date_range
        if venues is not None:
            search_params["venues"] = venues
        # Ensure fields contains title,abstract,year (Asta API needs these fields to match from abstract)
        if fields is not None:
            search_params["fields"] = fields
        else:
            search_params["fields"] = "title,abstract,year"
        search_params.update(search_kwargs)
        
        # 1. Search papers
        if mode == "query" or mode is None:
            papers = self.search_api.search_by_query(query, limit=limit, **search_params)
        elif mode == "title":
            paper = self.search_api.search_by_title(query, **search_params)
            papers = [paper] if paper else []
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
        if not papers:
            return []
        
        # 2. Apply minimum citation filter
        if self.min_citation is not None:
            papers = [p for p in papers 
                     if p.get("citation_counts", 0) >= self.min_citation]
        
        if not papers:
            return []
        
        # 3. If skip_reranking is True, return all results directly
        if skip_reranking:
            return papers
        
        # 4. If reranker is provided, perform reranking (using lazy loading)
        reranker = self._get_reranker()
        if reranker is not None and papers:
            papers = self._rerank_papers(query, papers)
        
        # 5. Return TopN
        return papers[:self.top_n]
    
    def retrieve_without_reranking(self, query: str, limit: int = 50,
                                  publication_date_range: Optional[str] = None,
                                  venues: Optional[str] = None,
                                  fields: Optional[str] = None,
                                  mode: Optional[str] = None,
                                  **search_kwargs) -> List[Dict[str, Any]]:
        """
        Retrieve papers without reranking (for global reranking scenarios)
        
        Args:
            query: Search query
            limit: Number of papers to return
            publication_date_range: Publication date range
            venues: Venue restrictions
            fields: Fields to return
            mode: Search mode
            **search_kwargs: Additional parameters
        
        Returns:
            List of papers without reranking
        """
        return self.retrieve(
            query=query,
            limit=limit,
            publication_date_range=publication_date_range,
            venues=venues,
            fields=fields,
            mode=mode,
            skip_reranking=True,
            **search_kwargs
        )
    
    def _get_reranker(self):
        """
        Get reranker instance (lazy loading to avoid pickle issues)
        
        Supports three modes:
        1. Direct mode: Use FlagReranker instance (if provided)
        2. Direct mode: Load FlagReranker from model path (lazy loading)
        3. API mode: Use RerankerEndpointPool (if API is configured)
        
        Returns:
            FlagReranker instance, RerankerEndpointPool instance, or None
        """
        # If reranker instance already exists, return it directly
        if self.reranker is not None:
            return self.reranker
        
        # If API mode is configured, use API
        if self._reranker_api_base_url or self._reranker_api_endpoint_pool_path:
            if self._reranker_endpoint_pool is None:
                try:
                    from shared.utils.reranker_endpoint_pool import RerankerEndpointPool
                    
                    if self._reranker_api_endpoint_pool_path:
                        # Load endpoint pool from file
                        self._reranker_endpoint_pool = RerankerEndpointPool(
                            pool_path=self._reranker_api_endpoint_pool_path
                        )
                    elif self._reranker_api_base_url:
                        # Use single base_url (could be load balancer address)
                        self._reranker_endpoint_pool = RerankerEndpointPool(
                            endpoints=[self._reranker_api_base_url]
                        )
                except ImportError:
                    print("Warning: RerankerEndpointPool not available, falling back to direct mode")
                    # Fall back to direct mode
                    pass
            
            if self._reranker_endpoint_pool is not None:
                return self._reranker_endpoint_pool
        
        # Direct mode: If model path is provided, try loading from pool (thread-safe, with global cache)
        if self._reranker_model_path:
            try:
                from shared.utils.reranker_pool import get_reranker
                # Get from pool (returns directly if already loaded, otherwise loads and caches)
                self.reranker = get_reranker(self._reranker_model_path, self._reranker_use_fp16)
                return self.reranker
            except ImportError:
                # If reranker_pool is not available, try direct loading (may have pickle issues)
                if FlagReranker:
                    self.reranker = FlagReranker(self._reranker_model_path, use_fp16=self._reranker_use_fp16)
                    return self.reranker
        
        return None
    
    def _rerank_papers(self, query: str, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rerank papers using BGE reranker (uses OpenScholar's rerank_paragraphs_bge function)
        
        Supports two modes:
        1. Direct mode: Use FlagReranker instance
        2. API mode: Use RerankerEndpointPool
        
        Args:
            query: Search query
            papers: List of papers
        
        Returns:
            Reranked list of papers
        """
        reranker = self._get_reranker()
        if reranker is None:
            return papers
        
        # Check if it's API mode (RerankerEndpointPool)
        from shared.utils.reranker_endpoint_pool import RerankerEndpointPool
        is_api_mode = isinstance(reranker, RerankerEndpointPool)
        
        # Use OpenScholar's rerank_paragraphs_bge function
        if is_api_mode:
            # API mode
            reranked_papers, result_dict, id_mapping = rerank_paragraphs_bge(
                query=query,
                paragraphs=papers,
                reranker=None,  # Not using direct mode
                reranker_endpoint_pool=reranker,  # Using API mode
                norm_cite=self.norm_cite,
                use_abstract=self.use_abstract,
                timeout=self._reranker_api_timeout,
            )
        else:
            # Direct mode
            reranked_papers, result_dict, id_mapping = rerank_paragraphs_bge(
                query=query,
                paragraphs=papers,
                reranker=reranker,  # Using direct mode
                reranker_endpoint_pool=None,  # Not using API mode
                norm_cite=self.norm_cite,
                use_abstract=self.use_abstract,
            )
        
        return reranked_papers
    
    @classmethod
    def create_with_semantic_scholar(cls,
                                    api_key: Optional[str] = None,
                                    reranker_model: Optional[str] = None,
                                    reranker: Optional[FlagReranker] = None,
                                    reranker_api_base_url: Optional[str] = None,
                                    reranker_api_endpoint_pool_path: Optional[str] = None,
                                    reranker_api_timeout: float = 30.0,
                                    top_n: int = 10,
                                    use_abstract: bool = False,
                                    norm_cite: bool = False,
                                    min_citation: Optional[int] = None) -> 'PaperRetriever':
        """
        Convenience method: Create PaperRetriever using Semantic Scholar API
        
        Args:
            api_key: Semantic Scholar API key
            reranker_model: Reranker model path (for lazy loading, avoids pickle issues)
            reranker: Optional pre-initialized reranker instance (if provided, will be used with priority)
            top_n: TopN count
            use_abstract: Whether to use abstract
            norm_cite: Whether to normalize citation counts
            min_citation: Minimum citation count
        """
        from .semantic_scholar_api import SemanticScholarAPI
        search_api = SemanticScholarAPI(api_key=api_key)
        
        # No longer create reranker instance here, use lazy loading
        
        return cls(
            search_api=search_api,
            reranker=reranker,  # May be None, will be obtained through lazy loading
            reranker_model_path=reranker_model,  # Save path for lazy loading
            reranker_use_fp16=True,
            reranker_api_base_url=reranker_api_base_url,
            reranker_api_endpoint_pool_path=reranker_api_endpoint_pool_path,
            reranker_api_timeout=reranker_api_timeout,
            top_n=top_n,
            use_abstract=use_abstract,
            norm_cite=norm_cite,
            min_citation=min_citation
        )
    
    @classmethod
    def create_with_asta(cls,
                         api_key: Optional[str] = None,
                         api_key_pool_path: Optional[str] = None,
                         endpoint: Optional[str] = None,
                         reranker_model: Optional[str] = None,
                         reranker: Optional[FlagReranker] = None,
                         reranker_api_base_url: Optional[str] = None,
                         reranker_api_endpoint_pool_path: Optional[str] = None,
                         reranker_api_timeout: float = 30.0,
                         top_n: int = 10,
                         use_abstract: bool = False,
                         norm_cite: bool = False,
                         min_citation: Optional[int] = None) -> 'PaperRetriever':
        """
        Convenience method: Create PaperRetriever using Asta API
        
        Args:
            api_key: Single Asta API key (backward compatible)
            api_key_pool_path: API key pool file path (priority higher than api_key)
            endpoint: MCP endpoint URL
            reranker_model: Reranker model path (for lazy loading, avoids pickle issues)
            reranker: Optional pre-initialized reranker instance (if provided, will be used with priority)
            top_n: TopN count
            use_abstract: Whether to use abstract
            norm_cite: Whether to normalize citation counts
            min_citation: Minimum citation count
        """
        from .asta_api import AstaAPI
        search_api = AstaAPI(
            api_key=api_key,
            api_key_pool_path=api_key_pool_path,
            endpoint=endpoint
        )
        
        # Note: No longer create reranker instance here
        # If reranker instance is provided, use it; otherwise save path for lazy loading
        # This avoids pickling reranker objects in multi-process environments
        
        return cls(
            search_api=search_api,
            reranker=reranker,  # May be None, will be obtained through lazy loading
            reranker_model_path=reranker_model,  # Save path for lazy loading
            reranker_use_fp16=True,
            reranker_api_base_url=reranker_api_base_url,
            reranker_api_endpoint_pool_path=reranker_api_endpoint_pool_path,
            reranker_api_timeout=reranker_api_timeout,
            top_n=top_n,
            use_abstract=use_abstract,
            norm_cite=norm_cite,
            min_citation=min_citation
        )

