"""
Paper Reviewer Agent

Complete paper reviewer that:
1. Uses RelatedWorkSearcher to find and summarize related work
2. Analyzes the paper
3. Produces comprehensive reviews
"""
import sys
from pathlib import Path
from typing import Dict, Optional, Any, List

# Add project root to path for shared utils
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from shared.utils.llm_service import LLMService, ChatMessage
from shared.utils.prompt_loader import get_prompt_loader
from shared.utils.json_parser import parse_review_json
from shared.utils.review_logger import ReviewLogger


class PaperReviewer:
    """
    Complete paper reviewer agent
    """
    
    def __init__(
        self,
        reviewer_llm_service: Optional[LLMService] = None,
        prompts_file: Optional[str] = None,
        logger: Optional[ReviewLogger] = None,
    ):
        """
        Initialize Paper Reviewer
        
        Args:
            reviewer_llm_service: LLM service for generating reviews (required)
            prompts_file: Optional path to prompts YAML file
            logger: Optional logger for logging
        """
        if reviewer_llm_service is None:
            raise ValueError("reviewer_llm_service is required")
        
        self.reviewer_llm_service = reviewer_llm_service
        self.prompt_loader = get_prompt_loader(prompts_file)
        self.logger = logger
        self.config = {}  # Will be set from config.yaml if available
    
    def review_paper(
        self,
        title: str,
        abstract: str,
        content: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        publication_date_range: Optional[str] = None,
        venues: Optional[str] = None,
        review_format: str = "detailed",
        auto_save_log: bool = True,
        verbose: bool = True,  # Control step output verbosity
    ) -> Dict[str, Any]:
        """
        Generate an initial review draft based solely on the paper content.
        This reviewer does not use external tools or information - it generates
        a review based on the paper itself.
        
        Args:
            title: Paper title
            abstract: Paper abstract
            content: Paper content (optional, can be full text or sections)
            keywords: Existing keywords (optional, for logging only)
            publication_date_range: Date range (optional, for logging only)
            venues: Venue filter (optional, for logging only)
            review_format: Review format ("detailed", "summary", "structured")
            
        Returns:
            Dictionary containing review sections (initial draft)
        """
        # Start logging run if logger is enabled
        run_id = None
        if self.logger:
            run_id = self.logger.start_run(
                title=title,
                abstract=abstract,
                content=content,
                keywords=keywords,
                publication_date_range=publication_date_range,
                venues=venues,
                review_format=review_format,
            )
            if run_id and verbose:
                print(f"Started review run with ID: {run_id}")
        
        # Generate initial review draft (without external tools)
        if verbose:
            print("=" * 60)
            print("Step 1: Reviewer generating initial review draft")
            print("=" * 60)
        
        # Build paper context
        paper_context = f"""Title: {title}

Abstract:
{abstract}"""

        if content:
            paper_context += f"""

Content:
{content[:8192]}"""  # Limit content to avoid token limits
        
        # Load review prompt from YAML
        review_prompt = self.prompt_loader.get_review_prompt(review_format=review_format)
        system_message = self.prompt_loader.get_reviewer_system_message()
        
        # Build prompt with only paper content (no external tools information)
        full_prompt = f"""{review_prompt}

Paper to review:
{paper_context}

Please provide a comprehensive review of this paper based on the content provided above."""

        # Log review prompt
        if self.logger:
            self.logger.log_review_prompt(full_prompt, system_message)

        messages = [
            ChatMessage(role="system", content=system_message),
            ChatMessage(role="user", content=full_prompt)
        ]
        
        # Get max_tokens from config if available, otherwise use default
        max_tokens = self.config.get('max_tokens', 16384) if hasattr(self, 'config') else 16384
        
        # Retry mechanism for review generation
        max_retries = 16
        last_error = None
        
        for attempt in range(max_retries):
            try:
                response = self.reviewer_llm_service.generate(
                    messages=messages,
                    temperature=0.7,
                    top_p=0.8,
                    max_tokens=max_tokens,
                )
                
                # Check if response is valid
                if response is None or not response.strip():
                    if attempt < max_retries - 1:
                        continue  # Retry silently
                    else:
                        last_error = "Received None or empty response from LLM"
                        break
                
                # Log LLM response
                if self.logger:
                    self.logger.log_review_llm_response(response)
                
                
                review_dict = parse_review_json(response, review_format=review_format)
                
                # If parsing succeeded, break out of retry loop
                if review_dict and ("review" in review_dict or len(review_dict) > 1):
                    # Ensure review field exists (store original markdown/text)
                    if "review" not in review_dict:
                        review_dict["review"] = response.strip()
                    
                    # Mark as markdown format if it contains markdown sections
                    if "## " in response or ("summary" in review_dict and len(review_dict) > 1):
                        review_dict["is_markdown"] = True
                    else:
                        review_dict["is_markdown"] = False
                    
                    # Log parsed review
                    if self.logger:
                        self.logger.log_parsed_review(review_dict)
                    
                    # Mark this as initial draft
                    review_dict["is_initial_draft"] = True
                    
                    # Add paper info
                    review_dict["title"] = title
                    review_dict["abstract"] = abstract
                    
                    # Log final output and save log (if auto_save_log is True)
                    if self.logger:
                        self.logger.log_final_output(review_dict)
                        # Automatically save the log at the end (unless called from refiner pipeline)
                        if auto_save_log:
                            log_path = self.logger.save_run()
                            if log_path and verbose:
                                print(f"Saved complete log to: {log_path}")
                    
                    return review_dict
                
                # If parsing failed, continue retrying silently
                if attempt == max_retries - 1:
                    last_error = "Failed to parse review from response"
                    
            except Exception as e:
                # Store the error, but don't print until all retries are exhausted
                last_error = e
        
        # All retries failed, output warning only once
        if last_error:
            error_msg = str(last_error)
            print(f"[WARN] Failed to generate review after {max_retries} attempts: {error_msg}")
        
        # Return error result
        error_result = {
            "error": str(last_error) if last_error else "Unknown error",
            "title": title,
            "abstract": abstract,
            "is_initial_draft": True,
        }
        # Log error
        if self.logger:
            self.logger.log_error(str(last_error) if last_error else "Unknown error", step="review_generation")
            self.logger.log_final_output(error_result)
            # Save log even on error
            if auto_save_log:
                log_path = self.logger.save_run()
                if log_path and verbose:
                    print(f"Saved log (with errors) to: {log_path}")
        return error_result
    
    def _parse_structured_review(self, review_text: str) -> Dict[str, Any]:
        """
        Parse structured review text into dictionary (fallback method)
        
        This is a fallback parser for when JSON parsing fails.
        The primary method is parse_review_json in utils.json_parser.
        """
        sections = {
            "summary": "",
            "strengths": [],
            "weaknesses": [],
            "comparison": "",
            "suggestions": [],
            "overall": "",
        }
        
        current_section = None
        lines = review_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check for section headers
            if line.startswith("## Summary") or "Summary" in line:
                current_section = "summary"
                continue
            elif line.startswith("## Strengths") or "Strengths" in line:
                current_section = "strengths"
                continue
            elif line.startswith("## Weaknesses") or "Weaknesses" in line:
                current_section = "weaknesses"
                continue
            elif "Comparison" in line or "Related Work" in line:
                current_section = "comparison"
                continue
            elif line.startswith("## Suggestions") or "Suggestions" in line:
                current_section = "suggestions"
                continue
            elif "Overall" in line or "Assessment" in line:
                current_section = "overall"
                continue
            
            # Add content to current section
            if current_section:
                if current_section in ["strengths", "weaknesses", "suggestions"]:
                    if line.startswith("-") or line.startswith("*"):
                        sections[current_section].append(line.lstrip("-* ").strip())
                    else:
                        sections[current_section].append(line)
                else:
                    if sections[current_section]:
                        sections[current_section] += " " + line
                    else:
                        sections[current_section] = line
        
        return {
            "review": review_text,
            "sections": sections,
        }

