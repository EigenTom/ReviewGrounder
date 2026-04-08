"""Review Refiner Agent - Refines review drafts with external tool information"""
import sys
import re
from pathlib import Path
from typing import Dict, Optional, Any

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from shared.utils.llm_service import LLMService, ChatMessage
from shared.utils.prompt_loader import get_prompt_loader
from shared.utils.json_parser import parse_review_json
from shared.utils.review_logger import ReviewLogger


class ReviewRefiner:
    """Agent for refining review drafts with external tool information"""
    
    def __init__(
        self,
        refiner_llm_service: Optional[LLMService] = None,
        prompts_file: Optional[str] = None,
        logger: Optional[ReviewLogger] = None,
    ):
        if refiner_llm_service is None:
            raise ValueError("refiner_llm_service is required")
        self.refiner_llm_service = refiner_llm_service
        self.prompt_loader = get_prompt_loader(prompts_file)
        self.logger = logger
        self.config = {}  # Will be set from config.yaml if available
    
    def refine_review(
        self,
        initial_review: Dict[str, Any],
        insight_miner_json: Optional[str] = None,
        results_analyzer_json: Optional[str] = None,
        related_work_json_list: Optional[str] = None,
        title: Optional[str] = None,
        abstract: Optional[str] = None,
        content: Optional[str] = None,
        review_format: str = "detailed",
        verbose: bool = True,  # Control step output verbosity
    ) -> Dict[str, Any]:
        """Refine an initial review draft using external tool information"""
        if verbose:
            print("=" * 60)
            print("Step 3: Refiner optimizing review with external tool information")
            print("=" * 60)
        
        initial_review_text = self._format_review_dict(initial_review, review_format) if isinstance(initial_review, dict) else str(initial_review)
        refiner_prompt = self.prompt_loader.get_refiner_prompt(review_format=review_format)
        system_message = self.prompt_loader.get_refiner_system_message()
        
        # Replace placeholders in the refiner prompt
        # The prompt uses <<paper_text>>, <<draft_review>>, <<insight_miner_json>>, <<results_analyzer_json>>, <<related_work_json_list>>
        context_parts = []
        if title:
            context_parts.append(f"Title: {title}")
        if abstract:
            context_parts.append(f"Abstract: {abstract}")
        if content:
            context_parts.append(f"Content: {content}")
        paper_text = "\n\n".join(context_parts) if context_parts else "Not provided"
        
        # Format draft review as text
        draft_review = initial_review_text
        
        # Format JSON inputs
        insight_miner_str = insight_miner_json if insight_miner_json else "{}"
        results_analyzer_str = results_analyzer_json if results_analyzer_json else "{}"
        related_work_json_str = related_work_json_list if related_work_json_list else "[]"
        
        # Replace placeholders in the prompt
        full_prompt = refiner_prompt.replace("<<paper_text>>", paper_text)
        full_prompt = full_prompt.replace("<<draft_review>>", draft_review)
        full_prompt = full_prompt.replace("<<insight_miner_json>>", insight_miner_str)
        full_prompt = full_prompt.replace("<<results_analyzer_json>>", results_analyzer_str)
        full_prompt = full_prompt.replace("<<related_work_json_list>>", related_work_json_str)

        if self.logger:
            self.logger.log_refiner_prompt(full_prompt, system_message)

        messages = [
            ChatMessage(role="system", content=system_message),
            ChatMessage(role="user", content=full_prompt)
        ]
        
        # Get max_tokens from config if available, otherwise use default
        max_tokens = self.config.get('max_tokens', 16384) if hasattr(self, 'config') else 16384
        
        # Retry mechanism for review refinement
        max_retries = 16
        last_error = None
        
        for attempt in range(max_retries):
            try:
                response = self.refiner_llm_service.generate(
                    messages=messages,
                    temperature=0.0,
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
                
                if self.logger:
                    self.logger.log_refiner_llm_response(response)
                
                # Parse the refiner's JSON output
                refined_review_json = parse_review_json(response, review_format=review_format)
                
                # If parsing succeeded, break out of retry loop
                if refined_review_json and (len(refined_review_json) > 1 or "review" in refined_review_json):
                    if self.logger:
                        self.logger.log_parsed_refined_review(refined_review_json)
                    
                    # 1. Keep the original JSON output from refiner
                    # Create a clean copy of the parsed JSON (remove any parsing artifacts)
                    original_json = {}
                    json_fields = ["summary", "soundness", "presentation", "contribution", 
                                  "strengths", "weaknesses", "questions", "rating", "confidence", "decision"]
                    for field in json_fields:
                        if field in refined_review_json:
                            original_json[field] = refined_review_json[field]
                    
                    # 2. Extract scores into a separate dict
                    scores = {}
                    score_fields = ["soundness", "presentation", "contribution", "rating", "confidence", "decision"]
                    for field in score_fields:
                        value = None  # default so 'value' is always defined before use
                        if field in refined_review_json:
                            value = refined_review_json[field]
                            # For decision field, always keep as string
                            if field == "decision":
                                scores[field] = str(value) if value is not None else None
                            else:
                                # For numeric fields, try to convert to float if it's a string
                                if isinstance(value, str):
                                    try:
                                        # Handle cases like "3.0" or "3 / 5" -> extract first number
                                        num_match = re.search(r'(\d+\.?\d*)', value)
                                        if num_match:
                                            value = float(num_match.group(1))
                                        else:
                                            value = None
                                    except (ValueError, AttributeError):
                                        value = None
                                # If already a number, keep it
                                elif isinstance(value, (int, float)):
                                    value = float(value)
                                else:
                                    value = None
                                scores[field] = value
                        else:
                            scores[field] = value
                    
                    # 3. Generate markdown review from JSON
                    markdown_review = self._generate_markdown_review(refined_review_json)
                    
                    # Build final output structure
                    refined_review = {
                        "review_json": original_json,
                        "review_markdown": markdown_review,
                        "review": markdown_review,
                        "scores": scores,
                    }
                    
                    # Include all original fields
                    for key, value in refined_review_json.items():
                        if key not in ["review", "review_json", "review_markdown", "scores"]:
                            refined_review[key] = value
                    
                    refined_review["is_initial_draft"] = False
                    refined_review["is_refined"] = True
                    refined_review["title"] = title or initial_review.get("title")
                    refined_review["abstract"] = abstract or initial_review.get("abstract")
                    
                    # Store tool outputs
                    if insight_miner_json:
                        refined_review["insight_miner_json"] = insight_miner_json
                    if results_analyzer_json:
                        refined_review["results_analyzer_json"] = results_analyzer_json
                    if related_work_json_list:
                        refined_review["related_work_json_list"] = related_work_json_list
                    
                    refined_review["initial_review"] = initial_review
                    
                    # Save initial scores and decision (before refinement) for evaluation
                    initial_scores = initial_review.get("initial_scores", {})
                    if initial_scores:
                        refined_review["initial_scores"] = initial_scores
                    
                    if self.logger:
                        self.logger.log_final_output(refined_review)
                    
                    return refined_review
                
                # If parsing failed, continue retrying silently
                if attempt == max_retries - 1:
                    last_error = "Failed to parse refined review from response"
                    
            except Exception as e:
                # Store the error, but don't print until all retries are exhausted
                last_error = e
        
        # All retries failed, output warning only once
        if last_error:
            error_msg = str(last_error)
            print(f"[WARN] Failed to refine review after {max_retries} attempts: {error_msg}")
        
        # Return error result
        error_result = {
            "error": str(last_error) if last_error else "Unknown error",
            "title": title or initial_review.get("title"),
            "abstract": abstract or initial_review.get("abstract"),
            "initial_review": initial_review,
            "is_refined": False,
        }
        if self.logger:
            self.logger.log_error(str(last_error) if last_error else "Unknown error", step="review_refinement")
            self.logger.log_final_output(error_result)
        return error_result
    
    def _format_review_dict(self, review_dict: Dict[str, Any], review_format: str) -> str:
        """Format review dictionary to text for prompt"""
        # If it's already in markdown format, return as-is
        if "review" in review_dict and isinstance(review_dict["review"], str):
            review_text = review_dict["review"]
            # Check if it's markdown format (has ## sections)
            if "## " in review_text or review_dict.get("is_markdown", False):
                return review_text
        
        # Otherwise, format from structured fields
        parts = []
        if "summary" in review_dict:
            parts.append(f"## Summary\n\n{review_dict['summary']}")
        if "soundness" in review_dict:
            soundness_val = review_dict.get("soundness", "")
            parts.append(f"## Soundness\n\n{soundness_val}")
        if "presentation" in review_dict:
            presentation_val = review_dict.get("presentation", "")
            parts.append(f"## Presentation\n\n{presentation_val}")
        if "contribution" in review_dict:
            contribution_val = review_dict.get("contribution", "")
            parts.append(f"## Contribution\n\n{contribution_val}")
        if "strengths" in review_dict:
            strengths = review_dict["strengths"]
            strengths_text = "\n".join(f"- {s}" for s in strengths) if isinstance(strengths, list) else strengths
            parts.append(f"## Strengths\n\n{strengths_text}")
        if "weaknesses" in review_dict:
            weaknesses = review_dict["weaknesses"]
            weaknesses_text = "\n".join(f"- {w}" for w in weaknesses) if isinstance(weaknesses, list) else weaknesses
            parts.append(f"## Weaknesses\n\n{weaknesses_text}")
        if "questions" in review_dict:
            questions = review_dict["questions"]
            questions_text = "\n".join(f"- {q}" for q in questions) if isinstance(questions, list) else questions
            parts.append(f"## Questions\n\n{questions_text}")
        if "rating" in review_dict:
            parts.append(f"## Rating\n\n{review_dict['rating']}")
        if "confidence" in review_dict:
            parts.append(f"## Confidence\n\n{review_dict['confidence']}")
        if "decision" in review_dict:
            parts.append(f"## Decision\n\n{review_dict['decision']}")
        if "comparison" in review_dict:
            parts.append(f"## Comparison with Related Work\n\n{review_dict['comparison']}")
        if "suggestions" in review_dict:
            suggestions = review_dict["suggestions"]
            suggestions_text = "\n".join(f"- {s}" for s in suggestions) if isinstance(suggestions, list) else suggestions
            parts.append(f"## Suggestions\n\n{suggestions_text}")
        if "overall_assessment" in review_dict:
            parts.append(f"## Overall Assessment\n\n{review_dict['overall_assessment']}")
        
        if parts:
            return "\n\n".join(parts)
        
        # Fallback: return review text if available
        # fix: in case it is not a string
        if type(review_dict.get("review", "")) != str:
            # use all its keys to assemble a review
            outlier_review_dict = review_dict.get("review", "")
            keys = outlier_review_dict.keys()
            print(f"[DEBUG] unconventional review dict, keys are: {keys}")
            
            parts = []
            for key in keys:
                this_part = outlier_review_dict[key]
                this_part_text = f"## {key.capitalize()}:\n\n{this_part}"
                parts.append(this_part_text)
                
            if parts:
                return "\n\n".join(parts)
            
            
        return str(review_dict.get("review", ""))
    
    def _generate_markdown_review(self, review_dict: Dict[str, Any]) -> str:
        """
        Generate natural language markdown review from structured JSON
        
        Format similar to pred_fast_mode_baseline:
        ## Summary:
        ...
        ## Soundness:
        ...
        ## Presentation:
        ...
        etc.
        """
        sections = []
        
        # Summary
        if "summary" in review_dict and review_dict["summary"]:
            sections.append(f"## Summary:\n\n{review_dict['summary']}")
        
        # Strengths
        if "strengths" in review_dict and review_dict["strengths"]:
            strengths = review_dict["strengths"]
            strengths_text = strengths if isinstance(strengths, str) else "\n\n".join(f"- {s}" for s in strengths)
            sections.append(f"## Strengths:\n\n{strengths_text}")
        
        # Weaknesses
        if "weaknesses" in review_dict and review_dict["weaknesses"]:
            weaknesses = review_dict["weaknesses"]
            weaknesses_text = weaknesses if isinstance(weaknesses, str) else "\n\n".join(f"- {w}" for w in weaknesses)
            sections.append(f"## Weaknesses:\n\n{weaknesses_text}")
        
        # Questions
        if "questions" in review_dict and review_dict["questions"]:
            questions = review_dict["questions"]
            questions_text = questions if isinstance(questions, str) else "\n\n".join(f"- {q}" for q in questions)
            sections.append(f"## Questions:\n\n{questions_text}")
        
        # Scoring sections - for JSON format, scores are already in the dict
        if "soundness" in review_dict:
            sections.append(f"## Soundness:\n\n{review_dict['soundness']}")
        
        if "presentation" in review_dict:
            sections.append(f"## Presentation:\n\n{review_dict['presentation']}")
        
        if "contribution" in review_dict:
            sections.append(f"## Contribution:\n\n{review_dict['contribution']}")
        
        if "rating" in review_dict:
            sections.append(f"## Rating:\n\n{review_dict['rating']}")
        
        if "confidence" in review_dict:
            sections.append(f"## Confidence:\n\n{review_dict['confidence']}")
        
        if "decision" in review_dict:
            sections.append(f"## Decision:\n\n{review_dict['decision']}")
        
        # If no sections were generated, return the raw review text if available
        if not sections and "review" in review_dict:
            return review_dict["review"]
        
        return "\n\n".join(sections)
