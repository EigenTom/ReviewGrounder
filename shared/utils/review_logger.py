"""
Review Logger Utility

Captures and logs all intermediate outputs from the review pipeline
"""
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List


class ReviewLogger:
    """
    Logger for capturing complete review pipeline execution logs
    """
    
    def __init__(self, log_dir: Optional[str] = None, enabled: bool = True):
        """
        Initialize Review Logger
        
        Args:
            log_dir: Directory to save log files. If None, uses current directory.
            enabled: Whether logging is enabled
        """
        self.enabled = enabled
        self.log_dir = Path(log_dir) if log_dir else Path.cwd()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Current run data
        self.current_run_id: Optional[str] = None
        self.current_run_data: Optional[Dict[str, Any]] = None
    
    def start_run(
        self,
        title: str,
        abstract: str,
        content: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        publication_date_range: Optional[str] = None,
        venues: Optional[str] = None,
        review_format: str = "detailed",
    ) -> str:
        """
        Start a new review run and generate UUID
        
        IMPORTANT: If current_run_data already exists, this method will preserve existing
        intermediate_outputs data to prevent data loss. Only input data and metadata are updated.
        
        Args:
            title: Paper title
            abstract: Paper abstract
            content: Paper content (optional)
            keywords: Existing keywords (optional)
            publication_date_range: Date range filter (optional)
            venues: Venue filter (optional)
            review_format: Review format
            
        Returns:
            Run UUID string
        """
        if not self.enabled:
            return ""
        
        # Generate UUID based on timestamp
        timestamp = datetime.now()
        # Use timestamp-based UUID (UUID1 uses MAC address + timestamp)
        run_id = str(uuid.uuid1())
        
        # PRESERVE existing intermediate_outputs if current_run_data already exists
        # This prevents data loss if start_run() is called multiple times
        existing_intermediate_outputs = None
        existing_final_output = None
        existing_errors = None
        if self.current_run_data is not None:
            existing_intermediate_outputs = self.current_run_data.get("intermediate_outputs")
            existing_final_output = self.current_run_data.get("final_output")
            existing_errors = self.current_run_data.get("errors", [])
        
        self.current_run_id = run_id
        
        # Initialize intermediate_outputs: use existing data if available, otherwise create new
        if existing_intermediate_outputs is not None:
            # Preserve existing intermediate outputs
            intermediate_outputs = existing_intermediate_outputs
            # Only initialize None fields if they don't exist
            if "generated_keywords" not in intermediate_outputs:
                intermediate_outputs["generated_keywords"] = None
            if "retrieved_papers" not in intermediate_outputs:
                intermediate_outputs["retrieved_papers"] = []
            if "paper_summaries" not in intermediate_outputs:
                intermediate_outputs["paper_summaries"] = []
            if "related_work_json_list" not in intermediate_outputs:
                intermediate_outputs["related_work_json_list"] = None
            if "paper_results_analyzer_output" not in intermediate_outputs:
                intermediate_outputs["paper_results_analyzer_output"] = None
            if "paper_insight_miner_output" not in intermediate_outputs:
                intermediate_outputs["paper_insight_miner_output"] = None
            if "review_prompt" not in intermediate_outputs:
                intermediate_outputs["review_prompt"] = None
            if "review_llm_response" not in intermediate_outputs:
                intermediate_outputs["review_llm_response"] = None
            if "parsed_review" not in intermediate_outputs:
                intermediate_outputs["parsed_review"] = None
            if "refiner_prompt" not in intermediate_outputs:
                intermediate_outputs["refiner_prompt"] = None
            if "refiner_llm_response" not in intermediate_outputs:
                intermediate_outputs["refiner_llm_response"] = None
            if "parsed_refined_review" not in intermediate_outputs:
                intermediate_outputs["parsed_refined_review"] = None
        else:
            # Create new intermediate_outputs structure
            intermediate_outputs = {
                "generated_keywords": None,
                "retrieved_papers": [],
                "paper_summaries": [],
                "related_work_json_list": None,
                "paper_results_analyzer_output": None,
                "paper_insight_miner_output": None,
                "review_prompt": None,
                "review_llm_response": None,
                "parsed_review": None,
                "refiner_prompt": None,
                "refiner_llm_response": None,
                "parsed_refined_review": None,
            }
        
        self.current_run_data = {
            "run_id": run_id,
            "timestamp": timestamp.isoformat(),
            "input": {
                "title": title,
                "abstract": abstract,
                "content": content,
                "keywords": keywords,
                "publication_date_range": publication_date_range,
                "venues": venues,
                "review_format": review_format,
            },
            "intermediate_outputs": intermediate_outputs,
            "final_output": existing_final_output,
            "errors": existing_errors if existing_errors is not None else [],
        }
        
        return run_id
    
    def log_keywords(self, keywords: List[str]):
        """Log generated search keywords"""
        if self.enabled and self.current_run_data:
            # Ensure intermediate_outputs exists
            if "intermediate_outputs" not in self.current_run_data:
                self.current_run_data["intermediate_outputs"] = {}
            self.current_run_data["intermediate_outputs"]["generated_keywords"] = keywords
    
    def log_retrieved_papers(self, papers: List[Dict[str, Any]]):
        """Log retrieved papers (raw)"""
        if self.enabled and self.current_run_data:
            # Ensure intermediate_outputs exists
            if "intermediate_outputs" not in self.current_run_data:
                self.current_run_data["intermediate_outputs"] = {}
            # Store paper metadata (may be large, so we store essential info)
            self.current_run_data["intermediate_outputs"]["retrieved_papers"] = [
                {
                    "paper_id": p.get("paper_id"),
                    "title": p.get("title"),
                    "authors": p.get("authors", [])[:10],  # Limit authors
                    "year": p.get("year"),
                    "venue": p.get("venue"),
                    "abstract": p.get("abstract", "")[:500],  # Truncate abstract
                    "citation_counts": p.get("citation_counts", 0),
                }
                for p in papers
            ]
    
    def log_paper_summary(self, paper_title: str, summary: str, paper_index: int):
        """Log a single paper summary"""
        if self.enabled and self.current_run_data:
            # Ensure intermediate_outputs exists
            if "intermediate_outputs" not in self.current_run_data:
                self.current_run_data["intermediate_outputs"] = {}
            if "paper_summaries" not in self.current_run_data["intermediate_outputs"]:
                self.current_run_data["intermediate_outputs"]["paper_summaries"] = []
            self.current_run_data["intermediate_outputs"]["paper_summaries"].append({
                "paper_index": paper_index,
                "paper_title": paper_title,
                "summary": summary,
            })
    
    def log_related_work_json_list(self, related_work_json_list: List[Dict[str, Any]]):
        """Log the final related work JSON list"""
        if self.enabled and self.current_run_data:
            # Ensure intermediate_outputs exists
            if "intermediate_outputs" not in self.current_run_data:
                self.current_run_data["intermediate_outputs"] = {}
            self.current_run_data["intermediate_outputs"]["related_work_json_list"] = related_work_json_list
    
    def log_paper_results_analyzer_output(self, results_analyzer_output: str):
        """Log the paper results analyzer JSON output"""
        if self.enabled and self.current_run_data:
            # Ensure intermediate_outputs exists
            if "intermediate_outputs" not in self.current_run_data:
                self.current_run_data["intermediate_outputs"] = {}
            self.current_run_data["intermediate_outputs"]["paper_results_analyzer_output"] = results_analyzer_output
    
    def log_paper_insight_miner_output(self, insight_miner_output: str):
        """Log the paper insight miner JSON output"""
        if self.enabled and self.current_run_data:
            # Ensure intermediate_outputs exists
            if "intermediate_outputs" not in self.current_run_data:
                self.current_run_data["intermediate_outputs"] = {}
            self.current_run_data["intermediate_outputs"]["paper_insight_miner_output"] = insight_miner_output
    
    def log_review_prompt(self, prompt: str, system_message: Optional[str] = None):
        """Log the review prompt sent to LLM"""
        if self.enabled and self.current_run_data:
            # Ensure intermediate_outputs exists
            if "intermediate_outputs" not in self.current_run_data:
                self.current_run_data["intermediate_outputs"] = {}
            self.current_run_data["intermediate_outputs"]["review_prompt"] = {
                "system_message": system_message,
                "user_prompt": prompt,
            }
    
    def log_review_llm_response(self, response: str):
        """Log the raw LLM response for review"""
        if self.enabled and self.current_run_data:
            # Ensure intermediate_outputs exists
            if "intermediate_outputs" not in self.current_run_data:
                self.current_run_data["intermediate_outputs"] = {}
            self.current_run_data["intermediate_outputs"]["review_llm_response"] = response
    
    def log_parsed_review(self, parsed_review: Dict[str, Any]):
        """Log the parsed review dictionary"""
        if self.enabled and self.current_run_data:
            # Ensure intermediate_outputs exists
            if "intermediate_outputs" not in self.current_run_data:
                self.current_run_data["intermediate_outputs"] = {}
            self.current_run_data["intermediate_outputs"]["parsed_review"] = parsed_review
    
    def log_refiner_prompt(self, prompt: str, system_message: Optional[str] = None):
        """Log the refiner prompt sent to LLM"""
        if self.enabled and self.current_run_data:
            # Ensure intermediate_outputs exists
            if "intermediate_outputs" not in self.current_run_data:
                self.current_run_data["intermediate_outputs"] = {}
            self.current_run_data["intermediate_outputs"]["refiner_prompt"] = {
                "system_message": system_message,
                "user_prompt": prompt,
            }
    
    def log_refiner_llm_response(self, response: str):
        """Log the raw LLM response for refiner"""
        if self.enabled and self.current_run_data:
            # Ensure intermediate_outputs exists
            if "intermediate_outputs" not in self.current_run_data:
                self.current_run_data["intermediate_outputs"] = {}
            self.current_run_data["intermediate_outputs"]["refiner_llm_response"] = response
    
    def log_parsed_refined_review(self, parsed_review: Dict[str, Any]):
        """Log the parsed refined review dictionary"""
        if self.enabled and self.current_run_data:
            # Ensure intermediate_outputs exists
            if "intermediate_outputs" not in self.current_run_data:
                self.current_run_data["intermediate_outputs"] = {}
            self.current_run_data["intermediate_outputs"]["parsed_refined_review"] = parsed_review
    
    def log_final_output(self, final_output: Dict[str, Any]):
        """Log the final review output"""
        if self.enabled and self.current_run_data:
            self.current_run_data["final_output"] = final_output
    
    def log_error(self, error: str, step: Optional[str] = None):
        """Log an error that occurred during execution"""
        if self.enabled and self.current_run_data:
            if "errors" not in self.current_run_data:
                self.current_run_data["errors"] = []
            self.current_run_data["errors"].append({
                "step": step,
                "error": error,
                "timestamp": datetime.now().isoformat(),
            })
    
    def save_run(self) -> Optional[str]:
        """
        Save the current run to a JSON file
        
        Returns:
            Path to saved log file, or None if logging is disabled
        """
        if not self.enabled or not self.current_run_data:
            return None
        
        # Generate filename with timestamp and UUID
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"review_log_{timestamp_str}_{self.current_run_id[:8]}.json"
        log_path = self.log_dir / filename
        
        # Save to JSON
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(self.current_run_data, f, indent=2, ensure_ascii=False)
        
        return str(log_path)
    
    def get_current_run_id(self) -> Optional[str]:
        """Get the current run ID"""
        return self.current_run_id if self.enabled else None
