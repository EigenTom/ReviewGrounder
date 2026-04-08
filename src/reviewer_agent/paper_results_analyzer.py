"""
Paper Results Analyzer Agent

This agent:
1. Reads paper content and candidate review
2. Analyzes the experiment/results parts of the candidate review
3. Provides JSON output with facts, review issues, and rewrite suggestions
"""
import sys
from pathlib import Path
from typing import List, Dict, Optional, Any

# Add project root to path for shared utils
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from shared.utils.llm_service import LLMService, ChatMessage
from shared.utils.prompt_loader import get_prompt_loader
from shared.utils.json_parser import parse_json_response
from shared.utils.review_logger import ReviewLogger

# Max retries when LLM output is not valid JSON (same as paper_reviewer / review_refiner)
MAX_JSON_RETRIES = 5

class PaperResultsAnalyzer:
    """
    Agent for analyzing paper experiment results and candidate reviews
    """
    
    def __init__(
        self,
        prompts_file: Optional[str] = None,
        llm_service: Optional[LLMService] = None,
        logger: Optional[ReviewLogger] = None,
    ):
        """
        Initialize Paper Results Analyzer
        
        Args:
            prompts_file: Optional path to prompts YAML file
            llm_service: LLM service for analyzing paper results (required)
            logger: Logger for logging
        """
        if llm_service is None:
            raise ValueError("llm_service is required")
        
        self.prompt_loader = get_prompt_loader(prompts_file)
        self.llm_service = llm_service
        self.logger = logger
        
    def analyze_paper_results(self, content: str, candidate_review: str) -> str:
        """
        Analyze the paper experiment results and candidate review
        
        Args:
            content: Content of the paper
            candidate_review: The candidate review draft to analyze
            
        Returns:
            JSON string with facts, review issues, and rewrite suggestions
        """
        prompt = self.prompt_loader.get_paper_results_analyzer_prompt(
            content=content,
            candidate_review=candidate_review
        )
        messages = [
            ChatMessage(role="user", content=prompt)
        ]
        
        fallback = '{"facts": {}, "review_issues": {}, "rewrite_suggestions": []}'
        last_error = None

        for attempt in range(MAX_JSON_RETRIES):
            try:
                response = self.llm_service.generate(
                    messages=messages,
                    temperature=0.7,
                    max_tokens=8192,
                )
            except Exception as e:
                last_error = e
                print(f"Error analyzing paper results (attempt {attempt + 1}/{MAX_JSON_RETRIES}): {e}")
                if self.logger:
                    self.logger.log_error(str(e), step="paper_results_analysis")
                if attempt == MAX_JSON_RETRIES - 1:
                    return fallback
                continue

            if response is None or not response.strip():
                if attempt == MAX_JSON_RETRIES - 1:
                    return fallback
                continue

            parsed = parse_json_response(response)
            if parsed is not None and isinstance(parsed, dict):
                if self.logger:
                    self.logger.log_paper_results_analyzer_output(response)
                return response

            last_error = "Output could not be parsed as JSON"
            
            # for llama3: if the output has content, directly return!
            if response.strip():
                print("[DEBUG] Result Analyzer's response has content but cannot be parsed as JSON, returning the content directly")
                return str(response)
            
            
            if attempt < MAX_JSON_RETRIES - 1:
                continue
            print(f"Results analyzer: failed to get valid JSON after {MAX_JSON_RETRIES} attempts, returning fallback")
            if self.logger:
                self.logger.log_error(str(last_error), step="paper_results_analysis")
                
            # can generate the content but cannot be parsed as json
            fallback = str(response)
            print(f"\n\n\n\n{fallback}\n\n\n\n")
                
            return fallback