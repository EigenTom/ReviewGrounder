"""
Utility for loading prompts from YAML configuration files
"""
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class PromptLoader:
    """Load and manage prompts from YAML files"""
    
    def __init__(self, prompts_file: Optional[str] = None):
        """
        Initialize prompt loader
        
        Args:
            prompts_file: Path to prompts YAML file. If None, uses default location.
        """
        if prompts_file is None:
            # Default to shared/configs/prompts.yaml relative to project root
            project_root = Path(__file__).parent.parent.parent
            prompts_file = project_root / "shared" / "configs" / "prompts.yaml"
        
        self.prompts_file = Path(prompts_file)
        self._prompts = None
        self._load_prompts()
    
    def _load_prompts(self):
        """Load prompts from YAML file"""
        if not self.prompts_file.exists():
            raise FileNotFoundError(f"Prompts file not found: {self.prompts_file}")
        
        with open(self.prompts_file, 'r', encoding='utf-8') as f:
            self._prompts = yaml.safe_load(f)
    
    def get_keyword_generation_prompt(self, context: str) -> str:
        """
        Get keyword generation prompt with context filled in
        
        Args:
            context: Paper information context
            
        Returns:
            Formatted prompt string
        """
        template = self._prompts["keyword_generation"]["user"]
        return template.format(context=context)
    
    def get_keyword_generation_system(self) -> str:
        """Get keyword generation system message"""
        return self._prompts["keyword_generation"].get("system", "")
    
    def get_paper_summarization_prompt(self, reference_paper: str, related_paper: str) -> str:
        """
        Get paper summarization prompt with reference_paper and related_paper filled in
        
        Args:
            reference_paper: Reference paper information (the paper being reviewed)
            related_paper: Related paper information
            
        Returns:
            Formatted prompt string
        """
        template = self._prompts["paper_summarization"]["user"]
        return template.format(reference_paper=reference_paper, related_paper=related_paper)
    
    def get_paper_results_summarization_prompt(self, content: str) -> str:
        """
        Get paper results summarization prompt with content filled in
        
        Args:
            content: Paper content (experiment results section)
            
        Returns:
            Formatted prompt string
        """
        template = self._prompts["paper_results_summarization"]["user"]
        return template.format(content=content)
    
    def get_paper_insight_miner_prompt(self, content: str, candidate_review: str) -> str:
        """
        Get paper insight miner prompt with content and candidate_review filled in
        
        Args:
            content: Paper content
            candidate_review: Candidate review draft
            
        Returns:
            Formatted prompt string
        """
        template = self._prompts["paper_insight_miner"]["user"]
        # Use replace instead of format to avoid issues with JSON braces in the template
        prompt = template.replace("{content}", content)
        prompt = prompt.replace("{candidate_review}", candidate_review)
        return prompt
    
    def get_paper_results_analyzer_prompt(self, content: str, candidate_review: str) -> str:
        """
        Get paper results analyzer prompt with content and candidate_review filled in
        
        Args:
            content: Paper content
            candidate_review: Candidate review draft
            
        Returns:
            Formatted prompt string
        """
        template = self._prompts["paper_results_analyzer"]["user"]
        # Use replace instead of format to avoid issues with JSON braces in the template
        prompt = template.replace("{content}", content)
        prompt = prompt.replace("{candidate_review}", candidate_review)
        return prompt
    
    def get_review_prompt(self, review_format: str = "detailed") -> str:
        """
        Get review prompt for specified format
        
        Args:
            review_format: Review format ("detailed", "summary", "structured")
            
        Returns:
            Review prompt string
        """
        if review_format not in self._prompts["review_prompts"]:
            review_format = "detailed"
        
        return self._prompts["review_prompts"][review_format]
    
    def get_reviewer_system_message(self) -> str:
        """Get system message for reviewer"""
        return self._prompts.get("reviewer_system", "You are an expert academic reviewer with deep knowledge in the field.")
    
    def get_refiner_prompt(self, review_format: str = "detailed") -> str:
        """
        Get refiner prompt for specified format
        
        Args:
            review_format: Review format ("detailed", "summary", "structured")
            
        Returns:
            Refiner prompt string
        """
        if "refiner_prompts" not in self._prompts:
            raise ValueError("refiner_prompts not found in prompts file")
        
        if review_format not in self._prompts["refiner_prompts"]:
            review_format = "detailed"
        
        return self._prompts["refiner_prompts"][review_format]
    
    def get_refiner_system_message(self) -> str:
        """Get system message for refiner"""
        return self._prompts.get("refiner_system", "You are an expert review refiner with deep knowledge in academic review quality standards and meta rubrics.")
    
    def get_rubrics_template(self) -> str:
        """
        Get the rubrics template for generating paper-specific rubrics.
        
        Returns:
            Rubrics template string (JSON array format)
        """
        return self._prompts.get("rubrics", "")
    
    def get_rubric_generation_prompt(self, version: str = "v2") -> str:
        """
        Get rubric generation prompt.
        
        Args:
            version: Prompt version ("v1" or "v2", default: "v2")
            
        Returns:
            Rubric generation prompt template string
        """
        key = f"{version}_rubric_generation_prompt"
        prompt = self._prompts.get(key, "")
        
        # For v2, replace rubric_template placeholder with actual template
        if version == "v2" and "<<rubric_template>>" in prompt:
            rubric_template = self.get_rubrics_template()
            prompt = prompt.replace("<<rubric_template>>", rubric_template)
        
        return prompt
    
    def get_evaluator_prompt(self, version: str = "v1") -> str:
        """
        Get evaluator prompt for evaluating reviews using rubrics.
        
        Args:
            version: Prompt version ("v0" or "v1", default: "v1")
            
        Returns:
            Evaluator prompt template string
        """
        key = f"{version}_evaluator_prompt"
        return self._prompts.get(key, "")
    
    def reload(self):
        """Reload prompts from file"""
        self._load_prompts()


# Global prompt loader instance
_prompt_loader: Optional[PromptLoader] = None


def get_prompt_loader(prompts_file: Optional[str] = None) -> PromptLoader:
    """
    Get or create global prompt loader instance
    
    Args:
        prompts_file: Optional path to prompts file
        
    Returns:
        PromptLoader instance
    """
    global _prompt_loader
    if _prompt_loader is None or prompts_file is not None:
        _prompt_loader = PromptLoader(prompts_file)
    return _prompt_loader

