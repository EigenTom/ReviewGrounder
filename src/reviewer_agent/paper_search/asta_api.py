"""
Asta API adapter (through Model Context Protocol)
"""
import os
import json
import time
import random
import requests
from typing import List, Dict, Optional, Any
from .paper_search_api import PaperSearchAPI

# try to import API Key Pool (if available)
try:
    from shared.utils.asta_api_key_pool import AstaAPIKeyPool
    HAS_KEY_POOL = True
except ImportError:
    AstaAPIKeyPool = None
    HAS_KEY_POOL = False


class AstaAPI(PaperSearchAPI):
    """Asta API implementation (through MCP protocol), supports API key pool and rotation"""
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 api_key_pool: Optional[AstaAPIKeyPool] = None,
                 api_key_pool_path: Optional[str] = None,
                 endpoint: Optional[str] = None,
                 max_retries: int = 3,
                 retry_delay: float = 1.0,
                 retry_backoff: float = 2.0):
        """
        initialize Asta API
        
        Args:
            api_key: single Asta API key (backward compatible, lowest priority)
            api_key_pool: AstaAPIKeyPool instance (highest priority)
            api_key_pool_path: API key pool file path (if provided, will create pool)
            endpoint: MCP endpoint URL, default is official endpoint
            max_retries: maximum retry attempts (when all keys fail)
            retry_delay: initial retry delay (seconds)
            retry_backoff: exponential backoff factor for retry delay
        """
        self.endpoint = endpoint or "https://asta-tools.allen.ai/mcp/v1"
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_backoff = retry_backoff
        
        # initialize API key pool
        if api_key_pool:
            self.key_pool = api_key_pool
        elif api_key_pool_path and HAS_KEY_POOL:
            self.key_pool = AstaAPIKeyPool(pool_path=api_key_pool_path)
        elif api_key:
            # single key mode (backward compatible)
            if HAS_KEY_POOL:
                self.key_pool = AstaAPIKeyPool(keys=[api_key])
            else:
                self.key_pool = None
                self.api_key = api_key
        elif HAS_KEY_POOL:
            # Try to create pool from environment variable (supports comma-separated keys)
            env_val = os.environ.get("ASTA_API_KEY")
            if env_val:
                keys = [k.strip() for k in env_val.split(",") if k.strip()]
                self.key_pool = AstaAPIKeyPool(keys=keys)
            else:
                raise ValueError(
                    "Asta API key is required. Provide api_key, api_key_pool, "
                    "api_key_pool_path, or set ASTA_API_KEY environment variable."
                )
        else:
            # Fallback single-key mode without pool: if multiple keys are provided,
            # pick the first one so behaviour is deterministic.
            env_val = os.environ.get("ASTA_API_KEY")
            self.api_key = None
            if env_val:
                self.api_key = env_val.split(",")[0].strip()
            if not self.api_key:
                raise ValueError(
                    "Asta API key is required. Set ASTA_API_KEY environment variable "
                    "or pass api_key parameter."
                )
        
        # set initial headers (will be updated when actual call is made)
        self._update_headers()
    
    def _update_headers(self, api_key: Optional[str] = None):
        """update request headers (use specified key or from pool)"""
        if self.key_pool:
            current_key = api_key or self.key_pool.get_key()
        else:
            current_key = api_key or self.api_key
        
        self.headers = {
            'x-api-key': current_key,
            'Content-Type': 'application/json',
            'Accept': 'application/json, text/event-stream',
            'User-Agent': 'AstaAPI-Python-Client/1.0'
        }
        return current_key
    
    def _call_mcp_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        call MCP tool (supports key rotation and debounce retry)
        
        Args:
            tool_name: tool name
            arguments: tool arguments
            
        Returns:
            API returned result, usually in {"content": ...} format
        """
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }
        
        # debounce retry strategy
        last_exception = None
        attempted_keys = set()  # record keys attempted in this round
        
        for retry_attempt in range(self.max_retries + 1):
            # get current API key
            current_key = None
            if self.key_pool:
                current_key = self.key_pool.get_key()
                # if this key has been attempted and there are other keys available, try skipping
                if current_key in attempted_keys and len(attempted_keys) < len(self.key_pool.keys):
                    # reset rotation, try other keys
                    self.key_pool.reset_round()
                    current_key = self.key_pool.get_key()
                    if current_key in attempted_keys:
                        # all keys have been attempted, wait and retry
                        if retry_attempt < self.max_retries:
                            delay = self.retry_delay * (self.retry_backoff ** retry_attempt)
                            time.sleep(delay)
                            self.key_pool.reset_round()
                            attempted_keys.clear()
                            current_key = self.key_pool.get_key()
            else:
                current_key = self.api_key
            
            # update headers
            self._update_headers(current_key)
            attempted_keys.add(current_key)
            
            try:
                response = requests.post(
                    self.endpoint,
                    json=payload,
                    headers=self.headers,
                    timeout=30
                )
                response.raise_for_status()
                
                # parse SSE format response
                response_text = response.text.strip()
                lines = response_text.split('\n')
                
                # find lines containing "data:"
                json_str = None
                for line in lines:
                    line = line.strip()
                    if line.startswith('data: '):
                        json_str = line[6:]  # skip "data: "
                        break
                
                if json_str is None:
                    json_str = response_text
                
                # parse JSON-RPC response
                rpc_response = json.loads(json_str)
                
                # check for errors
                if "error" in rpc_response:
                    error_msg = rpc_response.get("error", {}).get("message", str(rpc_response["error"]))
                    
                    # determine error type
                    error_code = rpc_response.get("error", {}).get("code", 0)
                    if error_code == 429 or "rate limit" in error_msg.lower() or "quota" in error_msg.lower():
                        error_type = "rate_limit"
                    elif error_code == 401 or "unauthorized" in error_msg.lower() or "invalid" in error_msg.lower():
                        error_type = "auth_error"
                    elif error_code >= 500:
                        error_type = "server_error"
                    else:
                        error_type = "other"
                    
                    # mark key error
                    if self.key_pool:
                        self.key_pool.mark_error(current_key, error_type)
                    
                    # if authentication error, this key may be permanently invalid, but continue trying other keys
                    if error_type == "auth_error":
                        if retry_attempt < self.max_retries:
                            continue  # try next key
                    
                    # if rate limit, wait and retry
                    if error_type == "rate_limit":
                        if retry_attempt < self.max_retries:
                            delay = self.retry_delay * (self.retry_backoff ** retry_attempt)
                            # add some randomness to avoid all clients retrying simultaneously
                            delay += random.uniform(0, delay * 0.3)
                            time.sleep(delay)
                            continue
                    
                    # other errors, raise directly
                    raise ValueError(f"Asta API error: {error_msg}")
                
                # success: mark key as successful
                if self.key_pool:
                    self.key_pool.mark_success(current_key)
                
                # process MCP response format: result.content is a list, each element corresponds to a paper
                if "result" in rpc_response:
                    result_data = rpc_response["result"]
                    if isinstance(result_data, dict) and "content" in result_data:
                        content_array = result_data["content"]
                        if isinstance(content_array, list):
                            papers = []
                            for content_item in content_array:
                                if isinstance(content_item, dict) and "text" in content_item:
                                    text_content = content_item["text"]
                                    if isinstance(text_content, str):
                                        try:
                                            paper_data = json.loads(text_content)
                                            if isinstance(paper_data, list):
                                                papers.extend(paper_data)
                                            elif isinstance(paper_data, dict):
                                                papers.append(paper_data)
                                        except json.JSONDecodeError:
                                            continue
                                elif isinstance(content_item, dict):
                                    papers.append(content_item)
                            
                            return {"content": papers}
                        return {"content": content_array}
                    return result_data
                
                return None
                
            except requests.exceptions.HTTPError as e:
                # HTTP error
                status_code = e.response.status_code if e.response else None
                if status_code == 429:
                    error_type = "rate_limit"
                elif status_code == 401:
                    error_type = "auth_error"
                elif status_code and status_code >= 500:
                    error_type = "server_error"
                else:
                    error_type = "other"
                
                if self.key_pool:
                    self.key_pool.mark_error(current_key, error_type)
                
                last_exception = e
                
                # rate limit or server error, wait and retry
                if error_type in ["rate_limit", "server_error"]:
                    if retry_attempt < self.max_retries:
                        delay = self.retry_delay * (self.retry_backoff ** retry_attempt)
                        delay += random.uniform(0, delay * 0.3)
                        time.sleep(delay)
                        continue
                
                # if authentication error, try next key
                if error_type == "auth_error" and retry_attempt < self.max_retries:
                    continue
                
            except (json.JSONDecodeError, requests.exceptions.RequestException) as e:
                last_exception = e
                # network error or parsing error, mark and retry
                if self.key_pool:
                    self.key_pool.mark_error(current_key, "other")
                
                if retry_attempt < self.max_retries:
                    delay = self.retry_delay * (self.retry_backoff ** retry_attempt)
                    delay += random.uniform(0, delay * 0.3)
                    time.sleep(delay)
                    continue
        
        # all retries failed
        if last_exception:
            error_msg = f"Error calling Asta API tool {tool_name} after {self.max_retries + 1} attempts"
            if self.key_pool:
                error_msg += f" (tried {len(attempted_keys)} keys)"
            raise RuntimeError(error_msg) from last_exception
        else:
            raise RuntimeError(f"Error calling Asta API tool {tool_name}: unknown error")
    
    def search_by_query(self, query: str, limit: int = 50, 
                       publication_date_range: Optional[str] = None,
                       venues: Optional[str] = None,
                       fields: Optional[str] = None,
                       **kwargs) -> List[Dict[str, Any]]:
        """
        search papers by query string (using search_papers_by_relevance)
        
        Args:
            query: search query string (corresponds to keyword parameter of API)
            limit: maximum number of papers to return, default is 50
            publication_date_range: publication date range, supports multiple formats:
                - "2019-03-05" - specific date
                - "2019-03" - specific month
                - "2019" - specific year
                - "2016-03-05:2020-06-06" - date range
                - "1981-08-25:" - from specific date
                - ":2015-01" - to specific date
                - "2015:2020" - year range
            venues: restricted venues, comma separated, e.g. "Nature,N. Engl. J. Med."
            fields: fields to return, comma separated. Optional fields include:
                abstract, authors, citations, fieldsOfStudy, isOpenAccess, 
                journal, publicationDate, references, tldr, url, venue, year
                default includes: title,abstract,authors,year,url,citations
            **kwargs: other optional parameters
        """
        # build arguments
        arguments = {
            "keyword": query,
            "limit": limit
        }
        
        # set fields parameter (if not specified, use default value)
        if fields is not None:
            arguments["fields"] = fields
        else:
            # default fields: must contain title,abstract,year to match from abstract
            # according to Asta API documentation, if abstract is not specified, only title will be searched
            arguments["fields"] = "title,abstract,year"
        
        # add publication_date_range (if provided)
        if publication_date_range:
            arguments["publication_date_range"] = publication_date_range
        
        # add venues (if provided)
        if venues:
            arguments["venues"] = venues
        
        # add other kwargs parameters
        arguments.update(kwargs)
        
        result = self._call_mcp_tool("search_papers_by_relevance", arguments)
        
        if not result or "content" not in result:
            return []
        
        content = result["content"]
        
        # content may be a list or a single dictionary
        if isinstance(content, list):
            papers = content
        elif isinstance(content, dict):
            # single paper, wrap in list
            papers = [content]
        else:
            return []
        
        # normalize each paper
        normalized_papers = []
        for paper in papers:
            if isinstance(paper, dict):
                normalized_papers.append(self._normalize_asta_paper(paper))
        
        return normalized_papers
    
    def search_by_title(self, title: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        search papers by title (using search_paper_by_title)
        """
        arguments = {
            "title": title,
            "fields": "title,abstract,authors,year,url,citations"
        }
        
        if "publication_date_range" in kwargs:
            arguments["publication_date_range"] = kwargs["publication_date_range"]
        if "venues" in kwargs:
            arguments["venues"] = kwargs["venues"]
        
        result = self._call_mcp_tool("search_paper_by_title", arguments)
        
        if not result or "content" not in result:
            return None
        
        content = result["content"]
        
        # content may be a list or a single dictionary
        if isinstance(content, list) and len(content) > 0:
            return self._normalize_asta_paper(content[0])
        elif isinstance(content, dict):
            return self._normalize_asta_paper(content)
        
        return None
    
    def get_paper(self, paper_id: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        get paper details by paper ID (using get_paper)
        
        Args:
            paper_id: can be various formats of ID (sha, CorpusId, DOI, ARXIV, etc.)
        """
        fields = kwargs.get("fields", "title,abstract,authors,year,url,citations")
        
        arguments = {
            "paper_id": paper_id,
            "fields": fields
        }
        
        result = self._call_mcp_tool("get_paper", arguments)
        
        if not result or "content" not in result:
            return None
        
        content = result["content"]
        
        if isinstance(content, dict):
            return self._normalize_asta_paper(content)
        
        return None
    
    def _normalize_asta_paper(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """
        normalize the paper format returned by Asta API
        
        supported fields:
        - title, abstract, authors, year, url, citations, venue, publicationDate
        - fieldsOfStudy, isOpenAccess, journal, references, tldr
        """
        # process authors field
        authors = []
        if "authors" in paper and paper["authors"]:
            if isinstance(paper["authors"], list):
                authors = [
                    author.get("name", "") if isinstance(author, dict) 
                    else str(author) 
                    for author in paper["authors"]
                ]
            elif isinstance(paper["authors"], str):
                # if authors is a string, try to split
                authors = [a.strip() for a in paper["authors"].split(",")]
        
        # get citation count
        citation_count = 0
        if "citations" in paper:
            if isinstance(paper["citations"], list):
                citation_count = len(paper["citations"])
            elif isinstance(paper["citations"], int):
                citation_count = paper["citations"]
            elif isinstance(paper["citations"], str):
                # try to convert to integer
                try:
                    citation_count = int(paper["citations"])
                except (ValueError, TypeError):
                    citation_count = 0
        
        # process year (may be from year or publicationDate)
        year = paper.get("year")
        if year is None and "publicationDate" in paper:
            pub_date = paper["publicationDate"]
            if isinstance(pub_date, str):
                # try to extract year from date string (format like "2020-03-15")
                try:
                    year = int(pub_date.split("-")[0])
                except (ValueError, IndexError, AttributeError):
                    pass
        
        # build normalized paper object
        normalized = {
            "title": paper.get("title", ""),
            "abstract": paper.get("abstract", "") or paper.get("tldr", ""),
            "text": paper.get("abstract", "") or paper.get("tldr", "") or "",  # text for reranking
            "url": paper.get("url", ""),
            "citation_counts": citation_count,
            "year": year,
            "authors": authors,
            "paper_id": paper.get("paperId", "") or paper.get("id", "") or paper.get("paper_id", ""),
            "venue": paper.get("venue", "") or paper.get("journal", ""),
            # extra fields (if exist)
            "publication_date": paper.get("publicationDate"),
            "fields_of_study": paper.get("fieldsOfStudy", []),
            "is_open_access": paper.get("isOpenAccess", False),
            "journal": paper.get("journal", ""),
            "tldr": paper.get("tldr", "")
        }
        return normalized
    
    def snippet_search(self, query: str, limit: int = 250, 
                      venues: Optional[str] = None,
                      paper_ids: Optional[str] = None,
                      inserted_before: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Asta specific snippet search functionality
        
        Args:
            query: search query
            limit: maximum number of snippets to return
            venues: restricted venues
            paper_ids: restricted paper IDs (comma separated)
            inserted_before: restricted insertion date
        """
        arguments = {
            "query": query,
            "limit": limit
        }
        
        if venues:
            arguments["venues"] = venues
        if paper_ids:
            arguments["paper_ids"] = paper_ids
        if inserted_before:
            arguments["inserted_before"] = inserted_before
        
        result = self._call_mcp_tool("snippet_search", arguments)
        
        if not result or "content" not in result:
            return []
        
        content = result["content"]
        
        if not isinstance(content, list):
            return []
        
        snippets = []
        for snippet in content:
            if isinstance(snippet, dict):
                normalized = self._normalize_asta_snippet(snippet)
                snippets.append(normalized)
        
        return snippets
    
    def _normalize_asta_snippet(self, snippet: Dict[str, Any]) -> Dict[str, Any]:
        """normalize Asta snippet format to paper format"""
        paper_info = snippet.get("paper", {})
        
        authors = []
        if "authors" in paper_info and paper_info["authors"]:
            if isinstance(paper_info["authors"], list):
                authors = [author.get("name", "") if isinstance(author, dict) else str(author) 
                           for author in paper_info["authors"]]
        
        normalized = {
            "title": paper_info.get("title", ""),
            "abstract": paper_info.get("abstract", ""),
            "text": snippet.get("text", "") or snippet.get("snippet", ""),  # snippet text content
            "url": paper_info.get("url", ""),
            "citation_counts": len(paper_info.get("citations", [])) if isinstance(paper_info.get("citations"), list) else 0,
            "year": paper_info.get("year"),
            "authors": authors,
            "paper_id": paper_info.get("paperId", "") or paper_info.get("id", ""),
            "venue": paper_info.get("venue", "")
        }
        return normalized

