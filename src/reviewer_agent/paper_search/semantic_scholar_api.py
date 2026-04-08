"""
Semantic Scholar API adapter
"""
import os
import requests
import time
from typing import List, Dict, Optional, Any
from .paper_search_api import PaperSearchAPI


class SemanticScholarAPI(PaperSearchAPI):
    """Semantic Scholar API implementation"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Semantic Scholar API
        
        Args:
            api_key: Semantic Scholar API key, if None, reads from S2_API_KEY environment variable
        """
        self.api_key = api_key or os.environ.get("S2_API_KEY")
        if not self.api_key:
            raise ValueError("Semantic Scholar API key is required. Set S2_API_KEY environment variable or pass api_key parameter.")
        
        self.base_url = "https://api.semanticscholar.org/graph/v1"
        self.headers = {'x-api-key': self.api_key}
        self.default_fields = 'title,year,abstract,url,authors.name,citationCount,year,externalIds'
    
    def search_by_query(self, query: str, limit: int = 50, min_citation_count: int = 0, 
                       sort: str = "citationCount:desc", **kwargs) -> List[Dict[str, Any]]:
        """
        Search papers by query string
        
        Args:
            query: Search query string
            limit: Maximum number of papers to return
            min_citation_count: Minimum citation count filter
            sort: Sort method, default is by citation count descending
            **kwargs: Other optional parameters
        """
        query_params = {
            'query': query,
            'limit': limit,
            'minCitationCount': min_citation_count,
            'sort': sort,
            'fields': self.default_fields
        }
        query_params.update(kwargs)
        
        try:
            response = requests.get(
                f'{self.base_url}/paper/search',
                params=query_params,
                headers=self.headers
            )
            time.sleep(0.5)  # Avoid too frequent requests
            
            if response.status_code == 200:
                response_data = response.json()
                if "data" in response_data and len(response_data["data"]) > 0:
                    papers = []
                    for paper in response_data["data"]:
                        normalized = self._normalize_s2_paper(paper)
                        papers.append(normalized)
                    return papers
            else:
                print(f"Semantic Scholar search failed with status code {response.status_code}: {response.text}")
                return []
        except Exception as e:
            print(f"Error searching Semantic Scholar: {e}")
            return []
    
    def search_by_title(self, title: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Search papers by title
        """
        query_params = {
            'query': title,
            'fields': self.default_fields
        }
        query_params.update(kwargs)
        
        try:
            response = requests.get(
                f'{self.base_url}/paper/search/match',
                params=query_params,
                headers=self.headers
            )
            time.sleep(0.2)
            
            if response.status_code == 200:
                response_data = response.json()
                if "data" in response_data and len(response_data["data"]) > 0:
                    return self._normalize_s2_paper(response_data["data"][0])
            return None
        except Exception as e:
            print(f"Error searching by title in Semantic Scholar: {e}")
            return None
    
    def get_paper(self, paper_id: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Get paper details by paper ID
        
        Args:
            paper_id: Can be Semantic Scholar ID, CorpusId, DOI, ARXIV, etc.
        """
        # Determine paper_id format
        if paper_id.isdigit():
            url = f'{self.base_url}/paper/CorpusID:{paper_id}'
        else:
            url = f'{self.base_url}/paper/{paper_id}'
        
        fields = kwargs.get('fields', self.default_fields)
        params = {'fields': fields}
        
        try:
            response = requests.get(url, params=params, headers=self.headers)
            time.sleep(0.1)
            
            if response.status_code == 200:
                paper = response.json()
                return self._normalize_s2_paper(paper)
            return None
        except Exception as e:
            print(f"Error getting paper from Semantic Scholar: {e}")
            return None
    
    def _normalize_s2_paper(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize Semantic Scholar paper format"""
        authors = []
        if "authors" in paper and paper["authors"]:
            authors = [author.get("name", "") for author in paper["authors"]]
        
        normalized = {
            "title": paper.get("title", ""),
            "abstract": paper.get("abstract", ""),
            "text": paper.get("abstract", ""),  # Default to using abstract as text
            "url": paper.get("url", ""),
            "citation_counts": paper.get("citationCount", 0),
            "year": paper.get("year"),
            "authors": authors,
            "paper_id": paper.get("paperId", ""),
            "venue": "",  # Semantic Scholar API doesn't directly return venue
            "externalIds": paper.get("externalIds", {})
        }
        return normalized

