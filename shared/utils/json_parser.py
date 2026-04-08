"""
Robust JSON parsing utilities for LLM responses
"""
import json
import re
from typing import Any, Dict, List, Optional


def extract_json_from_text(text: str) -> Optional[str]:
    """
    Extract JSON from text by removing markdown code block markers
    
    Args:
        text: Text that may contain JSON in markdown code blocks or plain JSON
        
    Returns:
        Extracted JSON string or None if not found
    """
    if not text:
        return None
    
    text_stripped = text.strip()
    
    # Try to parse as plain JSON first (no code blocks)
    try:
        json.loads(text_stripped)
        return text_stripped
    except json.JSONDecodeError:
        pass
    
    # Remove markdown code block markers: ```json ... ``` or ``` ... ```
    if text_stripped.startswith('```json'):
        # Remove ```json at start and ``` at end
        if text_stripped.endswith('```'):
            text_stripped = text_stripped[7:-3].strip()
        else:
            # No closing ```, just remove opening
            text_stripped = text_stripped[7:].strip()
    elif text_stripped.startswith('```'):
        # Handle ``` ... ``` (without json label)
        if text_stripped.endswith('```'):
            text_stripped = text_stripped[3:-3].strip()
        else:
            return None
    
    # Try to parse as JSON after removing code block markers
    try:
        json.loads(text_stripped)
        return text_stripped
    except json.JSONDecodeError:
        return None


def parse_json_response(text: str, fallback: Any = None) -> Any:
    """
    Parse JSON from LLM response with robust error handling
    
    Args:
        text: LLM response text
        fallback: Fallback value if parsing fails
        
    Returns:
        Parsed JSON object or fallback
    """
    if not text:
        return fallback
    
    # Extract JSON from text
    json_str = extract_json_from_text(text)
    
    if json_str is None:
        return fallback
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        # Try to fix common JSON issues
        json_str = fix_json_common_issues(json_str)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return fallback


def fix_json_common_issues(json_str: str) -> str:
    """
    Fix common JSON formatting issues
    
    Args:
        json_str: JSON string that may have issues
        
    Returns:
        Fixed JSON string
    """
    # Remove trailing commas
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*]', ']', json_str)
    
    # Fix single quotes to double quotes (basic)
    json_str = re.sub(r"'(\w+)':", r'"\1":', json_str)
    
    # Remove comments (basic)
    json_str = re.sub(r'//.*?$', '', json_str, flags=re.MULTILINE)
    json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
    
    return json_str


def parse_keywords_json(response: str) -> List[str]:
    """
    Parse keywords from JSON response
    
    Expected format:
    {"keywords": ["keyword1", "keyword2", ...]}
    or
    ["keyword1", "keyword2", ...]
    
    Args:
        response: LLM response text
        
    Returns:
        List of keywords, or empty list if parsing fails
    """
    if response is None:
        return []
    
    parsed = parse_json_response(response, fallback=None)
    
    if parsed is None:
        return []
    
    # Handle dict format: {"keywords": [...]}
    if isinstance(parsed, dict):
        if "keywords" in parsed and isinstance(parsed["keywords"], list):
            return parsed["keywords"][:5]
        return []
    
    # Handle list format: ["keyword1", "keyword2", ...]
    if isinstance(parsed, list):
        return parsed[:5]
    
    return []


def parse_summary_json(response: str) -> str:
    """
    Parse summary from JSON response
    
    Expected format:
    {"summary": "summary text"}
    or
    {"text": "summary text", "summary": "summary text"}
    
    Args:
        response: LLM response text
        
    Returns:
        Summary text
    """
    parsed = parse_json_response(response, fallback=None)
    
    if parsed is None:
        # Fallback to text parsing
        return response.strip()
    
    if isinstance(parsed, dict):
        # Try different possible keys
        for key in ["summary", "text", "content", "description"]:
            if key in parsed:
                summary = str(parsed[key]).strip()
                if summary:
                    return summary
    
    # Fallback to text parsing
    return response.strip()


def parse_review_json(response: str, review_format: str = "detailed") -> Dict[str, Any]:
    """
    Parse review from JSON or markdown response
    
    Expected formats:
    - JSON: {"summary": "...", "soundness": 5, ...}
    - Markdown: ## Summary\n\n...\n## Soundness\n\n...
    
    Args:
        response: LLM response text (JSON or markdown)
        review_format: Review format type (detailed, summary, structured)
        
    Returns:
        Review dictionary with parsed fields
    """
    # First try to parse as JSON
    parsed = parse_json_response(response, fallback=None)
    
    if parsed is not None and isinstance(parsed, dict):
        # JSON format - ensure it has required fields
        if "review" not in parsed:
            parsed["review"] = response.strip()
        return parsed
    
    # If not JSON, try to parse as markdown
    if "## " in response or "##" in response:
        markdown_parsed = parse_review_markdown(response)
        if len(markdown_parsed) > 1:  # More than just "review" field
            return markdown_parsed
    
    # Fallback to text parsing
    return {"review": response.strip()}


def parse_review_markdown(markdown_text: str) -> Dict[str, Any]:
    """
    Parse review from markdown format with sections like:
    ## Summary
    ...
    ## Soundness
    ...
    etc.
    
    Args:
        markdown_text: Markdown formatted review text
        
    Returns:
        Review dictionary with parsed fields
    """
    review_dict = {"review": markdown_text.strip()}
    
    # Pattern to match markdown sections: ## SectionName\n\ncontent
    section_pattern = r'##\s*([^\n]+)\s*\n\n(.*?)(?=\n##\s*|$)'
    matches = re.finditer(section_pattern, markdown_text, re.DOTALL)
    
    for match in matches:
        section_name = match.group(1).strip()
        section_content = match.group(2).strip()
        
        # Normalize section name (case-insensitive, remove extra spaces)
        section_name_lower = section_name.lower()
        
        # Map section names to dictionary keys
        if "summary" in section_name_lower:
            review_dict["summary"] = section_content
        elif "soundness" in section_name_lower:
            # Extract score - prioritize single float number (e.g., "3.0", "4.5")
            # If format is "3 / 5" or "**3 / 5**", extract the number before the slash
            score_val = None
            
            lines = section_content.split('\n')
            if lines:
                first_line = lines[0].strip()
                first_line_clean = re.sub(r'[`\*]', '', first_line)
                
                # Try to match number at start that's NOT followed by "/"
                num_match = re.match(r'^(\d+\.?\d*)(\s*)', first_line_clean)
                if num_match:
                    remaining = first_line_clean[len(num_match.group(0)):].strip()
                    if not remaining.startswith('/'):
                        try:
                            score_val = float(num_match.group(1))
                        except (ValueError, IndexError):
                            pass
                
                # If not found and there's a "/", try to extract number before "/" (e.g., "3 / 5" -> 3)
                if score_val is None and '/' in first_line_clean:
                    fraction_match = re.match(r'^\s*[`\*]*\s*(\d+\.?\d*)\s*[`\*]*\s*/\s*\d+', first_line_clean)
                    if fraction_match:
                        try:
                            score_val = float(fraction_match.group(1))
                        except (ValueError, IndexError):
                            pass
            
            # If not found, try to find number after "score:" or "rating:"
            if score_val is None:
                score_match = re.search(r'(?:score|rating)\s*[:=]\s*(\d+\.?\d*)', section_content, re.IGNORECASE)
                if score_match:
                    try:
                        score_val = float(score_match.group(1))
                    except (ValueError, IndexError):
                        pass
            
            if score_val is not None:
                review_dict["soundness"] = score_val  # Keep as float
        elif "presentation" in section_name_lower:
            score_val = None
            lines = section_content.split('\n')
            if lines:
                first_line = lines[0].strip()
                first_line_clean = re.sub(r'[`\*]', '', first_line)
                
                num_match = re.match(r'^(\d+\.?\d*)(\s*)', first_line_clean)
                if num_match:
                    remaining = first_line_clean[len(num_match.group(0)):].strip()
                    if not remaining.startswith('/'):
                        try:
                            score_val = float(num_match.group(1))
                        except (ValueError, IndexError):
                            pass
                
                if score_val is None and '/' in first_line_clean:
                    fraction_match = re.match(r'^\s*[`\*]*\s*(\d+\.?\d*)\s*[`\*]*\s*/\s*\d+', first_line_clean)
                    if fraction_match:
                        try:
                            score_val = float(fraction_match.group(1))
                        except (ValueError, IndexError):
                            pass
            
            if score_val is None:
                score_match = re.search(r'(?:score|rating)\s*[:=]\s*(\d+\.?\d*)', section_content, re.IGNORECASE)
                if score_match:
                    try:
                        score_val = float(score_match.group(1))
                    except (ValueError, IndexError):
                        pass
            
            if score_val is not None:
                review_dict["presentation"] = score_val
        elif "contribution" in section_name_lower:
            score_val = None
            lines = section_content.split('\n')
            if lines:
                first_line = lines[0].strip()
                first_line_clean = re.sub(r'[`\*]', '', first_line)
                
                num_match = re.match(r'^(\d+\.?\d*)(\s*)', first_line_clean)
                if num_match:
                    remaining = first_line_clean[len(num_match.group(0)):].strip()
                    if not remaining.startswith('/'):
                        try:
                            score_val = float(num_match.group(1))
                        except (ValueError, IndexError):
                            pass
                
                if score_val is None and '/' in first_line_clean:
                    fraction_match = re.match(r'^\s*[`\*]*\s*(\d+\.?\d*)\s*[`\*]*\s*/\s*\d+', first_line_clean)
                    if fraction_match:
                        try:
                            score_val = float(fraction_match.group(1))
                        except (ValueError, IndexError):
                            pass
            
            if score_val is None:
                score_match = re.search(r'(?:score|rating)\s*[:=]\s*(\d+\.?\d*)', section_content, re.IGNORECASE)
                if score_match:
                    try:
                        score_val = float(score_match.group(1))
                    except (ValueError, IndexError):
                        pass
            
            if score_val is not None:
                review_dict["contribution"] = score_val
        elif "strength" in section_name_lower:
            review_dict["strengths"] = section_content
        elif "weakness" in section_name_lower:
            review_dict["weaknesses"] = section_content
        elif "question" in section_name_lower:
            review_dict["questions"] = section_content
        elif "rating" in section_name_lower and "confidence" not in section_name_lower:
            score_val = None
            lines = section_content.split('\n')
            if lines:
                first_line = lines[0].strip()
                first_line_clean = re.sub(r'[`\*]', '', first_line)
                
                num_match = re.match(r'^(\d+\.?\d*)(\s*)', first_line_clean)
                if num_match:
                    remaining = first_line_clean[len(num_match.group(0)):].strip()
                    if not remaining.startswith('/'):
                        try:
                            score_val = float(num_match.group(1))
                        except (ValueError, IndexError):
                            pass
                
                if score_val is None and '/' in first_line_clean:
                    fraction_match = re.match(r'^\s*[`\*]*\s*(\d+\.?\d*)\s*[`\*]*\s*/\s*\d+', first_line_clean)
                    if fraction_match:
                        try:
                            score_val = float(fraction_match.group(1))
                        except (ValueError, IndexError):
                            pass
            
            if score_val is None:
                score_match = re.search(r'(?:score|rating)\s*[:=]\s*(\d+\.?\d*)', section_content, re.IGNORECASE)
                if score_match:
                    try:
                        score_val = float(score_match.group(1))
                    except (ValueError, IndexError):
                        pass
            
            if score_val is not None:
                review_dict["rating"] = score_val
        elif "confidence" in section_name_lower:
            score_val = None
            lines = section_content.split('\n')
            if lines:
                first_line = lines[0].strip()
                first_line_clean = re.sub(r'[`\*]', '', first_line)
                
                num_match = re.match(r'^(\d+\.?\d*)(\s*)', first_line_clean)
                if num_match:
                    remaining = first_line_clean[len(num_match.group(0)):].strip()
                    if not remaining.startswith('/'):
                        try:
                            score_val = float(num_match.group(1))
                        except (ValueError, IndexError):
                            pass
                
                if score_val is None and '/' in first_line_clean:
                    fraction_match = re.match(r'^\s*[`\*]*\s*(\d+\.?\d*)\s*[`\*]*\s*/\s*\d+', first_line_clean)
                    if fraction_match:
                        try:
                            score_val = float(fraction_match.group(1))
                        except (ValueError, IndexError):
                            pass
            
            if score_val is None:
                score_match = re.search(r'(?:score|rating)\s*[:=]\s*(\d+\.?\d*)', section_content, re.IGNORECASE)
                if score_match:
                    try:
                        score_val = float(score_match.group(1))
                    except (ValueError, IndexError):
                        pass
            
            if score_val is not None:
                review_dict["confidence"] = score_val
        elif "decision" in section_name_lower:
            review_dict["decision"] = section_content
    
    return review_dict
