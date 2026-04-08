"""
Related Work Searcher Agent

This agent:
1. Reads paper (title, abstract, content, keywords)
2. Generates search keywords from the paper
3. Calls Asta API to get related papers
4. Summarizes each related paper
5. Forms a related work text
"""
import sys
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path for shared utils
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from shared.utils.llm_service import LLMService, ChatMessage
from shared.utils.prompt_loader import get_prompt_loader
from shared.utils.json_parser import parse_keywords_json, parse_summary_json, parse_json_response
import json
from shared.utils.review_logger import ReviewLogger
from shared.utils.reranker import rerank_paragraphs_bge
from .paper_search import PaperRetriever


class RelatedWorkSearcher:
    """
    Agent for searching and summarizing related work
    """
    
    def __init__(
        self,
        paper_retriever: PaperRetriever,
        max_related_papers: int = 10,
        max_parallel_summaries: Optional[int] = None,
        prompts_file: Optional[str] = None,
        keyword_llm_service: Optional[LLMService] = None,
        summarizer_llm_service: Optional[LLMService] = None,
        logger: Optional[ReviewLogger] = None,
        verbose: bool = True,  # Set to False to suppress intermediate progress output
    ):
        """
        Initialize Related Work Searcher
        
        Args:
            paper_retriever: Paper retriever for searching papers
            max_related_papers: Maximum number of related papers to retrieve
            max_parallel_summaries: Maximum number of parallel paper summarization workers (default: 8)
            prompts_file: Optional path to prompts YAML file
            keyword_llm_service: LLM service for generating keywords (required)
            summarizer_llm_service: LLM service for summarizing papers (required)
        """
        if keyword_llm_service is None or summarizer_llm_service is None:
            raise ValueError("keyword_llm_service and summarizer_llm_service are required")
        
        self.keyword_llm_service = keyword_llm_service
        self.summarizer_llm_service = summarizer_llm_service
        self.paper_retriever = paper_retriever
        self.max_related_papers = max_related_papers
        self.max_parallel_summaries = max_parallel_summaries if max_parallel_summaries is not None else 8  # Not used anymore, kept for compatibility
        self.prompt_loader = get_prompt_loader(prompts_file)
        self.logger = logger
        self.verbose = verbose  # Control verbosity of intermediate output
        # Keep track of latest generated search keywords for UI/debug purposes.
        self.last_keywords: Optional[List[str]] = None
    
    def generate_search_keywords(
        self,
        title: str,
        abstract: str,
        content: Optional[str] = None,
        keywords: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Generate search keywords from paper information
        
        Args:
            title: Paper title
            abstract: Paper abstract
            content: Paper content (optional)
            keywords: Existing keywords (optional)
            
        Returns:
            List of search keywords
        """
        # Build context
        context_parts = [f"Title: {title}", f"Abstract: {abstract}"]
        if keywords:
            context_parts.append(f"Keywords: {', '.join(keywords)}")
        if content:
            # Use first 2000 characters of content if available
            context_parts.append(f"Content (excerpt): {content[:2000]}")
        
        context = "\n\n".join(context_parts)
        
        # Load prompt from YAML
        prompt = self.prompt_loader.get_keyword_generation_prompt(context=context)
        system_msg = self.prompt_loader.get_keyword_generation_system()
        
        messages = []
        if system_msg:
            messages.append(ChatMessage(role="system", content=system_msg))
        messages.append(ChatMessage(role="user", content=prompt))
        
        # Retry up to 16 times if JSON parsing fails
        max_retries = 16
        last_error = None
        
        for attempt in range(max_retries):
            try:
                response = self.keyword_llm_service.generate(
                    messages=messages,
                    temperature=0.7,
                    max_tokens=200,
                )
                
                # Check if response is None or empty
                if response is None:
                    print(f"[WARN] Received None response from LLM when generating keywords!!!")
                    if attempt < max_retries - 1:
                        continue  # Retry silently
                    else:
                        last_error = "Received None response from LLM"
                        break
                
                if not response.strip():
                    if attempt < max_retries - 1:
                        continue  # Retry silently
                    else:
                        last_error = "Received empty response from LLM"
                        break
                
                # Parse keywords from JSON response
                keywords = parse_keywords_json(response)
                
                # If parsing succeeded and we got keywords, return them
                if keywords:
                    # Log and cache keywords
                    if self.logger:
                        self.logger.log_keywords(keywords)
                    self.last_keywords = keywords[:5]
                    return keywords[:5]  # Return up to 5 keywords
                
                # If parsing returned empty list, check if response is valid JSON
                # If valid JSON but empty keywords, use fallback immediately (don't retry)
                import json
                try:
                    parsed = json.loads(response.strip())
                    # Valid JSON with keywords field but empty list - use fallback
                    if isinstance(parsed, dict) and "keywords" in parsed:
                        last_error = "LLM returned valid JSON but empty keywords list"
                        break
                except json.JSONDecodeError:
                    # Invalid JSON - continue retrying (fall through to retry logic below)
                    pass
                
                # If parsing failed (invalid JSON), add error message and retry (silently)
                if attempt < max_retries - 1:
                    # Add error feedback to prompt for next attempt
                    response_preview = response[:200] if response else "None"
                    error_msg = ChatMessage(
                        role="user",
                        content=f"The previous response was not valid JSON. Please respond with valid JSON only in this format:\n{{\n  \"keywords\": [\"keyword1\", \"keyword2\", \"keyword3\", \"keyword4\", \"keyword5\"]\n}}\n\nPrevious invalid response: {response_preview}..."
                    )
                    messages.append(error_msg)
                else:
                    last_error = "Failed to parse JSON from response after all retries"
                    break
                    
            except Exception as e:
                # Store the error, but don't print until all retries are exhausted
                last_error = e
        
        # All retries failed, output warning only once
        if last_error:
            print(f"[WARN] Failed to generate keywords after {max_retries} attempts: {last_error}, using fallback")
            # Log the error
            if self.logger:
                self.logger.log_error(f"Keyword generation failed: {last_error}", step="keyword_generation")
        
        # Fallback: extract simple keywords from title and abstract
        text = f"{title} {abstract}".lower()
        words = text.split()
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can'}
        keywords = [w for w in words if len(w) > 4 and w not in common_words]
        fallback_keywords = list(set(keywords))[:5]
        
        # Log fallback keywords
        if self.logger:
            self.logger.log_keywords(fallback_keywords)
        
        self.last_keywords = fallback_keywords
        return fallback_keywords
    
    def search_related_papers(
        self,
        keywords: List[str],
        publication_date_range: Optional[str] = None,
        venues: Optional[str] = None,
        limit_per_keyword: int = 20,
        min_year: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for related papers using keywords (without per-keyword reranking)
        
        Args:
            keywords: List of search keywords
            publication_date_range: Date range filter (optional, e.g., "2023:")
            venues: Venue filter (optional)
            limit_per_keyword: Number of papers to retrieve per keyword
            min_year: Minimum publication year (additional filter, applied after retrieval)
            
        Returns:
            List of related papers (not reranked, ready for global reranking)
        """
        all_papers = []
        seen_paper_ids = set()
        
        for keyword in keywords:
            try:
                # Use retrieve_without_reranking to skip per-keyword reranking
                # This allows us to collect all papers and do global reranking later
                papers = self.paper_retriever.retrieve_without_reranking(
                    query=keyword,
                    limit=limit_per_keyword,
                    publication_date_range=publication_date_range,
                    venues=venues,
                    mode="query",
                    fields="title,abstract,year",  # Ensure fields include abstract for matching
                )
                
                # Deduplicate by paper_id
                for paper in papers:
                    paper_id = paper.get("paper_id") or paper.get("title", "")
                    if paper_id and paper_id not in seen_paper_ids:
                        seen_paper_ids.add(paper_id)
                        all_papers.append(paper)
                
            except Exception as e:
                print(f"Error searching for keyword '{keyword}': {e}")
                continue
        
        # Apply year filtering (additional safeguard)
        if min_year is not None:
            filtered_papers = []
            for paper in all_papers:
                year = paper.get("year")
                if year is not None:
                    try:
                        year_int = int(year) if isinstance(year, str) else year
                        if year_int >= min_year:
                            filtered_papers.append(paper)
                    except (ValueError, TypeError):
                        # If year parsing fails, keep the paper (conservative approach)
                        filtered_papers.append(paper)
                else:
                    # If year is missing, keep the paper (conservative approach)
                    filtered_papers.append(paper)
            all_papers = filtered_papers
            if self.verbose:
                print(f"After year filtering (>= {min_year}): {len(all_papers)} papers")
        
        return all_papers
    
    def summarize_paper(
        self, 
        paper: Dict[str, Any],
        reference_title: Optional[str] = None,
        reference_abstract: Optional[str] = None,
        reference_content: Optional[str] = None,
    ) -> str:
        """
        Summarize a single paper with reference to the paper being reviewed
        
        Args:
            paper: Paper dictionary with title, abstract, etc.
            reference_title: Title of the reference paper (the paper being reviewed)
            reference_abstract: Abstract of the reference paper
            reference_content: Content of the reference paper (optional)
            
        Returns:
            Summary of the paper
        """
        title = paper.get("title", "Unknown")
        abstract = paper.get("abstract", "")
        authors = ", ".join(paper.get("authors", []))[:100]  # Limit author list
        year = paper.get("year", "Unknown")
        venue = paper.get("venue", "")
        
        # Get related paper content if available (try content, then text)
        related_content = paper.get("content") or paper.get("text") or ""
        
        # Build related paper information
        related_paper_parts = [
            f"Title: {title}",
            f"Authors: {authors}",
            f"Year: {year}",
            f"Venue: {venue}",
            f"Abstract: {abstract}"
        ]
        
        # Add content if available (limit to 3000 characters to avoid token limits)
        if related_content:
            related_paper_parts.append(f"Content (excerpt): {related_content[:3000]}")
        
        related_paper = "\n".join(related_paper_parts)
        
        # Build reference paper information
        reference_paper_parts = []
        if reference_title:
            reference_paper_parts.append(f"Title: {reference_title}")
        if reference_abstract:
            reference_paper_parts.append(f"Abstract: {reference_abstract}")
        if reference_content:
            # Use first 3000 characters of content if available
            reference_paper_parts.append(f"Content (excerpt): {reference_content[:3000]}")
        
        reference_paper = "\n".join(reference_paper_parts) if reference_paper_parts else "Title: Unknown\nAbstract: Not provided"

        # Load prompt from YAML with reference_paper and related_paper
        prompt = self.prompt_loader.get_paper_summarization_prompt(
            reference_paper=reference_paper,
            related_paper=related_paper
        )

        messages = [
            ChatMessage(role="user", content=prompt)
        ]
        
        # Retry mechanism for paper summarization
        max_retries = 16
        last_error = None
        
        for attempt in range(max_retries):
            try:
                response = self.summarizer_llm_service.generate(
                    messages=messages,
                    temperature=0.7,
                    max_tokens=800,  # Increased for structured JSON response
                )
                
                # Check if response is valid
                if response is None or not response.strip():
                    if attempt < max_retries - 1:
                        continue  # Retry silently
                    else:
                        last_error = "Received None or empty response from LLM"
                        break
                
                # Parse structured summary from JSON response
                parsed = parse_json_response(response, fallback=None)
                
                if parsed and isinstance(parsed, dict):
                    # Build formatted summary from structured fields, preserving structure
                    summary_parts = []
                    
                    # Add summary field if available
                    if parsed.get("summary"):
                        summary_parts.append(str(parsed["summary"]))
                    
                    # Add main methods if available (with clear label)
                    if parsed.get("main_methods"):
                        summary_parts.append(f"\nMain methods: {parsed['main_methods']}")
                    
                    # Add key findings if available (with clear label)
                    if parsed.get("key_findings"):
                        summary_parts.append(f"\nKey findings: {parsed['key_findings']}")
                    
                    # Add relation if available (with clear label)
                    if parsed.get("relation"):
                        summary_parts.append(f"\nRelation: {parsed['relation']}")
                    
                    if summary_parts:
                        # Join with newlines to preserve structure, but remove extra newlines
                        summary = "\n".join(summary_parts).strip()
                    else:
                        # Fallback to simple summary parsing
                        summary = parse_summary_json(response)
                elif parsed and isinstance(parsed, list):
                    # Handle case where LLM returns a list instead of dict
                    summary = parse_summary_json(response)
                else:
                    # Fallback to simple summary parsing
                    summary = parse_summary_json(response)
                
                # If we got a valid summary, return it
                if summary and len(summary) >= 10:
                    return summary.strip()
                
                # If summary is too short, continue retrying silently
                if attempt == max_retries - 1:
                    last_error = "Summary too short or empty"
                
            except Exception as e:
                # Store the error, but don't print until all retries are exhausted
                last_error = e
        
        # All retries failed, output warning only once
        if last_error:
            print(f"[WARN] Failed to summarize paper '{title}' after {max_retries} attempts: {last_error}, using fallback")
        
        # Fallback: use abstract if available
        return abstract[:300] if abstract else f"{title} ({year})"
    
    def _rerank_papers_globally(
        self,
        papers: List[Dict[str, Any]],
        query: str,
    ) -> List[Dict[str, Any]]:
        """
        Globally rerank papers using the given query (e.g., paper title + abstract)
        
        This is the key improvement: instead of reranking per keyword, we rerank
        all collected papers using the full paper context as query.
        
        Args:
            papers: List of papers to rerank
            query: Query string for reranking (typically title + abstract of the paper under review)
            
        Returns:
            Reranked list of papers
        """
        if not papers:
            return []
        
        # Get reranker (may trigger lazy loading)
        reranker = self.paper_retriever._get_reranker()
        if reranker is None:
            print("Warning: No reranker available, skipping global reranking")
            return papers
        
        if self.verbose:
            print(f"Globally reranking {len(papers)} papers using query: {query[:100]}...")
        
        # Check if it's API mode (RerankerEndpointPool) or direct mode (FlagReranker)
        try:
            from shared.utils.reranker_endpoint_pool import RerankerEndpointPool
            is_api_mode = isinstance(reranker, RerankerEndpointPool)
        except ImportError:
            is_api_mode = False
        
        # Use the rerank_paragraphs_bge function (same as OpenScholar)
        if is_api_mode:
            # API mode
            reranked_papers, result_dict, id_mapping = rerank_paragraphs_bge(
                query=query,
                paragraphs=papers,
                reranker=None,  # Not using direct mode
                reranker_endpoint_pool=reranker,  # Using API mode
                norm_cite=self.paper_retriever.norm_cite,
                use_abstract=self.paper_retriever.use_abstract,
                timeout=getattr(self.paper_retriever, '_reranker_api_timeout', 30.0),
            )
        else:
            # Direct mode
            reranked_papers, result_dict, id_mapping = rerank_paragraphs_bge(
                query=query,
                paragraphs=papers,
                reranker=reranker,  # Using direct mode
                reranker_endpoint_pool=None,  # Not using API mode
                norm_cite=self.paper_retriever.norm_cite,
                use_abstract=self.paper_retriever.use_abstract,
            )
        
        if self.verbose:
            print(f"Global reranking completed. Top 3 scores: {list(result_dict.values())[:3] if result_dict else 'N/A'}")
        
        return reranked_papers
    
    def generate_related_work_json_list(
        self,
        title: str,
        abstract: str,
        content: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        publication_date_range: Optional[str] = None,
        venues: Optional[str] = None,
        min_year: Optional[int] = 2023,
    ) -> List[Dict[str, Any]]:
        """
        Generate related work summaries as a JSON list (each item is a JSON dict)
        
        Returns structured JSON list of summaries.
        
        Args:
            title: Paper title
            abstract: Paper abstract
            content: Paper content (optional)
            keywords: Existing keywords (optional)
            publication_date_range: Date range filter (optional, e.g., "2023:")
            venues: Venue filter (optional)
            min_year: Minimum publication year (default: 2023)
            
        Returns:
            List of JSON dictionaries, each containing summary, main_methods, key_findings, relation
        """
        # Step 1: Generate search keywords
        if self.verbose:
            print("Generating search keywords...")
        search_keywords = self.generate_search_keywords(
            title=title,
            abstract=abstract,
            content=content,
            keywords=keywords,
        )
        if self.verbose:
            print(f"Generated keywords: {search_keywords}")
        
        # Step 2: Search for related papers
        if publication_date_range is None and min_year is not None:
            publication_date_range = f"{min_year}:"
        
        related_papers = self.search_related_papers(
            keywords=search_keywords,
            publication_date_range=publication_date_range,
            venues=venues,
            min_year=min_year,
        )
        if self.verbose:
            print(f"Found {len(related_papers)} related papers (after deduplication and year filtering)")
        
        # Log retrieved papers
        if self.logger:
            self.logger.log_retrieved_papers(related_papers)
        
        if not related_papers:
            return []
        
        # Step 3: Global reranking
        if self.verbose:
            print("Performing global reranking using paper title + abstract...")
        
        global_query = f"{title}\n{abstract}"
        # global_query = f"Represent this sentence for searching relevant passages: {title}\n{abstract}"
        
        related_papers = self._rerank_papers_globally(
            papers=related_papers,
            query=global_query,
        )
        
        # Step 4: Limit to top K papers
        related_papers = related_papers[:self.max_related_papers]
        if self.verbose:
            print(f"Selected top {len(related_papers)} papers after global reranking")
        
        # Step 5: Summarize each paper and collect JSON summaries
        if self.verbose:
            print(f"Summarizing {len(related_papers)} papers in parallel (max 10 concurrent requests)...")
        
        json_summaries = [None] * len(related_papers)
        completed_count = 0
        
        max_workers = 10
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(
                    self._summarize_paper_as_json,
                    idx,
                    paper,
                    title,
                    abstract,
                    content,
                    len(related_papers),
                ): idx
                for idx, paper in enumerate(related_papers)
            }
            
            for future in as_completed(future_to_idx):
                try:
                    idx, json_summary = future.result()
                    json_summaries[idx] = json_summary
                    completed_count += 1
                    if self.verbose:
                        print(f"Completed {completed_count}/{len(related_papers)} JSON summaries")
                except Exception as e:
                    idx = future_to_idx[future]
                    paper = related_papers[idx]
                    paper_title = paper.get("title", "Unknown")
                    print(f"Error processing paper '{paper_title}' (index {idx}): {e}")
                    import traceback
                    traceback.print_exc()
                    # Fallback: create minimal JSON summary
                    json_summaries[idx] = {
                        "summary": paper.get("abstract", "")[:300] if paper.get("abstract") else paper_title,
                        "main_methods": "",
                        "key_findings": "",
                        "relation": ""
                    }
                    completed_count += 1
                    if self.verbose:
                        print(f"Completed {completed_count}/{len(related_papers)} JSON summaries (with fallback)")
        
        # Filter out None values
        json_summaries = [s for s in json_summaries if s is not None]
        
        # Log the final JSON list
        if self.logger:
            if hasattr(self.logger, 'log_related_work_json_list'):
                self.logger.log_related_work_json_list(json_summaries)
        
        return json_summaries
    
    def _summarize_paper_as_json(
        self,
        idx: int,
        paper: Dict[str, Any],
        reference_title: str,
        reference_abstract: str,
        reference_content: Optional[str],
        total_papers: int,
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Helper function to summarize a single paper and return as JSON dict.
        Used for parallel processing in generate_related_work_json_list.
        
        Args:
            idx: Index of the paper (0-based)
            paper: Paper dictionary
            reference_title: Title of the reference paper
            reference_abstract: Abstract of the reference paper
            reference_content: Content of the reference paper (optional)
            total_papers: Total number of papers being processed
            
        Returns:
            Tuple of (index, json_summary_dict)
        """
        paper_title = paper.get("title", "Unknown")
        if self.verbose:
            print(f"[{idx + 1}/{total_papers}] Summarizing as JSON: {paper_title[:50]}...")
        
        try:
            # Get the raw JSON response from summarize_paper's internal logic
            title = paper.get("title", "Unknown")
            abstract = paper.get("abstract", "")
            authors = ", ".join(paper.get("authors", []))[:100]
            year = paper.get("year", "Unknown")
            venue = paper.get("venue", "")
            
            related_content = paper.get("content") or paper.get("text") or ""
            
            related_paper_parts = [
                f"Title: {title}",
                f"Authors: {authors}",
                f"Year: {year}",
                f"Venue: {venue}",
                f"Abstract: {abstract}"
            ]
            
            if related_content:
                related_paper_parts.append(f"Content (excerpt): {related_content[:3000]}")
            
            related_paper = "\n".join(related_paper_parts)
            
            reference_paper_parts = []
            if reference_title:
                reference_paper_parts.append(f"Title: {reference_title}")
            if reference_abstract:
                reference_paper_parts.append(f"Abstract: {reference_abstract}")
            if reference_content:
                reference_paper_parts.append(f"Content (excerpt): {reference_content[:3000]}")
            
            reference_paper = "\n".join(reference_paper_parts) if reference_paper_parts else "Title: Unknown\nAbstract: Not provided"
            
            prompt = self.prompt_loader.get_paper_summarization_prompt(
                reference_paper=reference_paper,
                related_paper=related_paper
            )
            
            messages = [
                ChatMessage(role="user", content=prompt)
            ]
            
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    response = self.summarizer_llm_service.generate(
                        messages=messages,
                        temperature=0.7,
                        max_tokens=800,
                    )
                    
                    if response is None or not response.strip():
                        if attempt < max_retries - 1:
                            continue
                        else:
                            raise ValueError("Received None or empty response from LLM")
                    
                    # Parse JSON response directly
                    parsed = parse_json_response(response, fallback=None)
                    
                    if parsed and isinstance(parsed, dict):
                        # Ensure all required fields are present
                        json_summary = {
                            "summary": parsed.get("summary", ""),
                            "main_methods": parsed.get("main_methods", ""),
                            "key_findings": parsed.get("key_findings", ""),
                            "relation": parsed.get("relation", "")
                        }
                        
                        if json_summary["summary"]:  # At least summary should be non-empty
                            return (idx, json_summary)
                    else:
                        # Fallback: try to extract summary from text
                        summary_text = parse_summary_json(response)
                        if summary_text and len(summary_text) >= 10:
                            json_summary = {
                                "summary": summary_text,
                                "main_methods": "",
                                "key_findings": "",
                                "relation": ""
                            }
                            return (idx, json_summary)
                    
                    if attempt == max_retries - 1:
                        raise ValueError("Failed to parse valid JSON summary")
                        
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    continue
            
            # Should not reach here, but just in case
            raise ValueError("Failed to get valid summary after retries")
            
        except Exception as e:
            print(f"Error summarizing paper '{paper_title}' as JSON: {e}")
            import traceback
            traceback.print_exc()
            # Fallback: create minimal JSON summary
            fallback_summary = paper.get("abstract", "")[:300] if paper.get("abstract") else paper_title
            json_summary = {
                "summary": fallback_summary,
                "main_methods": "",
                "key_findings": "",
                "relation": ""
            }
            return (idx, json_summary)

