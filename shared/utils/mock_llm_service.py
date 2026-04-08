"""
Mock LLM Service that returns pre-generated reviews from a JSON file
This is a hack for testing the refiner pipeline with existing reviews
"""
import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Any, Union

from .llm_service import LLMService, ChatMessage


def extract_title_from_latex(paper_context: str) -> Optional[str]:
    """Extract title from LaTeX format \\title{...}"""
    match = re.search(r'\\title\{([^}]+)\}', paper_context)
    if match:
        return match.group(1).strip()
    return None


def extract_abstract_from_latex(paper_context: str) -> Optional[str]:
    """Extract abstract from LaTeX format \\begin{abstract}...\\end{abstract}"""
    match = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', paper_context, re.DOTALL)
    if match:
        abstract = match.group(1).strip()
        # Clean up LaTeX commands
        abstract = re.sub(r'\\[a-zA-Z]+\{([^}]+)\}', r'\1', abstract)  # Remove LaTeX commands
        abstract = re.sub(r'\$([^$]+)\$', r'\1', abstract)  # Remove math mode
        return abstract
    return None


class MockLLMService(LLMService):
    """
    Mock LLM Service that returns pre-generated reviews from a JSON file
    
    This service matches papers by extracting title and abstract from paper_context
    and returns the corresponding pred_fast_mode_baseline from the JSON file.
    """
    
    def __init__(self, json_file_path: str):
        """
        Initialize Mock LLM Service
        
        Args:
            json_file_path: Path to JSON file containing pre-generated reviews
        """
        self.json_file_path = Path(json_file_path)
        if not self.json_file_path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_file_path}")
        
        # Load JSON data
        with open(self.json_file_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # Build index for faster lookup
        self._build_index()
    
    def _build_index(self):
        """Build index mapping (title, abstract) to review"""
        self.index = {}
        self.entries = []  # Store full entries for fallback matching
        self.initial_scores_index = {}  # Store initial scores and decision for each entry
        
        for entry in self.data:
            paper_context = entry.get('paper_context', '')
            title = extract_title_from_latex(paper_context)
            abstract = extract_abstract_from_latex(paper_context)
            
            # Extract review content (prefer meta_review.content, fallback to raw_text)
            model_prediction = entry.get('model_prediction', {})
            meta_review = model_prediction.get('meta_review', {})
            review_content = meta_review.get('content', '') or model_prediction.get('raw_text', '')
            
            # Extract initial scores and decision
            initial_scores = {
                'rating': meta_review.get('rating'),
                'soundness': meta_review.get('soundness'),
                'presentation': meta_review.get('presentation'),
                'contribution': meta_review.get('contribution'),
                'decision': model_prediction.get('decision'),
            }
            
            if title and abstract:
                # Use normalized title and first 200 chars of abstract as key
                normalized_title = title.lower().strip()
                normalized_abstract = abstract[:200].lower().strip()
                key = (normalized_title, normalized_abstract)
                self.index[key] = review_content
                self.initial_scores_index[key] = initial_scores
            
            
            # Store entry for fallback matching
            self.entries.append({
                'title': title,
                'abstract': abstract,
                'paper_context': paper_context,
                'review': review_content,
                'id': entry.get('id', ''),
                'initial_scores': initial_scores,
            })
    
    def _find_entry(self, messages: List[Union[ChatMessage, Dict[str, str]]]) -> Optional[Dict[str, Any]]:
        """
        Find entry by matching title and abstract from messages
        
        Args:
            messages: List of chat messages
            
        Returns:
            Entry dict with 'review' and 'initial_scores' or None if not found
        """
        # Extract paper context from user message
        user_message = None
        for msg in messages:
            if isinstance(msg, dict):
                if msg.get('role') == 'user':
                    user_message = msg.get('content', '')
            elif isinstance(msg, ChatMessage):
                if msg.role == 'user':
                    user_message = msg.content
        
        if not user_message:
            return None
        
        # Try to extract title and abstract from user message
        # Look for patterns like "Title: ..." or "Abstract: ..."
        title_match = re.search(r'Title:\s*(.+?)(?:\n|$)', user_message, re.IGNORECASE)
        abstract_match = re.search(r'Abstract:\s*(.+?)(?:\n\n|Content:|$)', user_message, re.DOTALL | re.IGNORECASE)
        
        extracted_title = None
        extracted_abstract = None
        
        if title_match and abstract_match:
            extracted_title = title_match.group(1).strip()
            extracted_abstract = abstract_match.group(1).strip()
        else:
            # Fallback: search in paper_context if available
            paper_context_match = re.search(r'Paper to review:\s*(.+?)(?:Please provide|$)', user_message, re.DOTALL)
            if paper_context_match:
                paper_context = paper_context_match.group(1)
                extracted_title = extract_title_from_latex(paper_context)
                extracted_abstract = extract_abstract_from_latex(paper_context)
        
        if extracted_title and extracted_abstract:
            # Normalize for matching
            normalized_title = extracted_title.lower().strip()
            normalized_abstract = extracted_abstract[:200].lower().strip()
            
            # Try exact match first
            key = (normalized_title, normalized_abstract)
            if key in self.index:
                return {
                    'review': self.index[key],
                    'initial_scores': self.initial_scores_index.get(key, {})
                }
            
            # Try fuzzy match (check if title matches)
            for (index_title, index_abstract), review in self.index.items():
                # Check title similarity (either contains or is contained)
                title_similar = (
                    normalized_title in index_title or 
                    index_title in normalized_title or
                    normalized_title == index_title
                )
                
                # Check abstract similarity (first 100 chars)
                abstract_similar = (
                    normalized_abstract[:100] in index_abstract[:100] or 
                    index_abstract[:100] in normalized_abstract[:100] or
                    normalized_abstract[:100] == index_abstract[:100]
                )
                
                if title_similar and abstract_similar:
                    return {
                        'review': review,
                        'initial_scores': self.initial_scores_index.get((index_title, index_abstract), {})
                    }
        
        # Final fallback: try to match by paper_context in entries
        for entry in self.entries:
            if entry['paper_context']:
                # Check if user message contains similar content
                entry_title = entry['title']
                if entry_title and extracted_title:
                    if entry_title.lower().strip() in extracted_title.lower() or extracted_title.lower() in entry_title.lower():
                        return {
                            'review': entry['review'],
                            'initial_scores': entry.get('initial_scores', {})
                        }
        
        return None
    
    def _find_review(self, messages: List[Union[ChatMessage, Dict[str, str]]]) -> Optional[str]:
        """
        Find review by matching title and abstract from messages
        
        Args:
            messages: List of chat messages
            
        Returns:
            Review text or None if not found
        """
        entry = self._find_entry(messages)
        if entry:
            return entry['review']
        return None
    
    def get_initial_scores(self, messages: List[Union[ChatMessage, Dict[str, str]]]) -> Optional[Dict[str, Any]]:
        """
        Get initial scores and decision by matching title and abstract from messages
        
        Args:
            messages: List of chat messages
            
        Returns:
            Dict with initial scores (rating, soundness, presentation, contribution, decision) or None if not found
        """
        entry = self._find_entry(messages)
        if entry:
            return entry.get('initial_scores', {})
        return None
    
    def generate(
        self,
        messages: List[Union[ChatMessage, Dict[str, str]]],
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        max_tokens: int = 16384,
        presence_penalty: float = 0.0,
        **kwargs
    ) -> str:
        """
        Generate text from messages (returns pre-generated review)
        
        Args:
            messages: List of chat messages
            temperature: Ignored (for compatibility)
            top_p: Ignored (for compatibility)
            top_k: Ignored (for compatibility)
            max_tokens: Ignored (for compatibility)
            presence_penalty: Ignored (for compatibility)
            **kwargs: Additional parameters (ignored)
            
        Returns:
            Pre-generated review text
        """
        review = self._find_review(messages)
        if review:
            return review
        
        # Fallback: return a default message
        return "## Summary:\n\nReview not found in pre-generated data."
    
    def stream_generate(
        self,
        messages: List[Union[ChatMessage, Dict[str, str]]],
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        max_tokens: int = 16384,
        presence_penalty: float = 0.0,
        **kwargs
    ):
        """
        Stream generate text from messages (yields pre-generated review)
        
        Yields:
            Pre-generated review text chunks
        """
        review = self._find_review(messages)
        if review:
            # Yield in chunks to simulate streaming
            chunk_size = 100
            for i in range(0, len(review), chunk_size):
                yield review[i:i + chunk_size]
        else:
            yield "## Summary:\n\nReview not found in pre-generated data."

