"""
Abstract base class for universal paper search API interface
Defines a unified interface for easy integration of different paper search services
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any


class PaperSearchAPI(ABC):
    """Abstract base class for paper search API"""
    
    @abstractmethod
    def search_by_query(self, query: str, limit: int = 50, **kwargs) -> List[Dict[str, Any]]:
        """
        Search papers by query string
        
        Args:
            query: Search query string
            limit: Maximum number of papers to return
            **kwargs: Other optional parameters (e.g., date range, venue, etc.)
            
        Returns:
            List of papers, each paper is a dictionary containing the following fields:
            - title: Paper title
            - abstract: Abstract
            - text: Text content (for reranking, usually abstract or passage)
            - url: Paper URL
            - citation_counts: Citation count
            - year: Publication year
            - authors: List of authors
            - paper_id: Unique paper identifier
            - venue: Publication venue (optional)
        """
        pass
    
    @abstractmethod
    def search_by_title(self, title: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Search papers by title
        
        Args:
            title: Paper title
            **kwargs: Other optional parameters
            
        Returns:
            Paper dictionary, returns None if not found
        """
        pass
    
    @abstractmethod
    def get_paper(self, paper_id: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Get paper details by paper ID
        
        Args:
            paper_id: Unique paper identifier
            **kwargs: Other optional parameters (e.g., fields to return)
            
        Returns:
            Paper dictionary, returns None if not found
        """
        pass
    
    def normalize_paper(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize paper formats returned by different APIs into a unified format
        
        Args:
            paper: Raw paper data
            
        Returns:
            Normalized paper dictionary
        """
        normalized = {
            "title": paper.get("title", ""),
            "abstract": paper.get("abstract", ""),
            "text": paper.get("text") or paper.get("abstract", ""),  # Text for reranking
            "url": paper.get("url", ""),
            "citation_counts": paper.get("citation_counts") or paper.get("citationCount") or 0,
            "year": paper.get("year"),
            "authors": paper.get("authors", []),
            "paper_id": paper.get("paper_id") or paper.get("paperId") or paper.get("id", ""),
            "venue": paper.get("venue", ""),
        }
        return normalized

