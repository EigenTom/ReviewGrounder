"""
Generate review-based rubrics by querying LLMs with concurrent parallel requests.

This script:
1. Reads the JSON file with review data
2. Extracts entries with 'id', 'pred_fast_mode_baseline', 'paper_context', and 'decision'
3. Loads the rubric generation prompt from prompts.yaml
4. Loads LLM configuration from configs.yaml (supports gpt and vllm modes)
5. For each entry, generates rubrics by replacing <<golden_review>> with the ground truth review
6. Uses concurrent parallel requests (ThreadPoolExecutor) for efficient LLM queries
7. Extracts rubrics from LLM responses and saves to eval_rubrics.json

Output JSON file (eval_rubrics.json) contains a list of dicts with:
- id: Entry identifier
- paper_context: Paper content
- decision: Decision field from input
- golden_review: The pred_fast_mode_baseline review (ground truth)
- rubrics: List of rubric objects, each with title, description, and weight

Usage:
    python 1_generate_review_based_rubrics.py \
        --json_path input.json \
        --output_path eval_rubrics.json \
        --yaml_path prompts.yaml \
        --config_path configs.yaml \
        --max_workers 5

The configs.yaml should specify either "gpt" or "vllm" mode and corresponding settings.
"""
import json
import os
import sys
import argparse
import yaml
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd
from dotenv import load_dotenv

# Add parent directory to path to import llm_service
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Import parse_llm_response from local llm_service module (for parsing LLM responses)
import llm_service as local_llm_service
parse_llm_response = local_llm_service.parse_llm_response

# Import from shared/utils for gpt/vllm support
# Add project root to path to enable absolute imports from shared.utils
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Use absolute imports from shared.utils package
from shared.utils.llm_service import LLMService
from shared.utils.vllm_service import VLLMService
from shared.utils.gpt_service import GPTService

# Load environment variables
load_dotenv()

class ReviewProcessor:
    """Handles the extraction and processing of reviews from different sources."""

    @staticmethod
    def extract_review_content(pred_context):
        """
        Extract the review content from the prediction context.

        Args:
            pred_context: Raw prediction data that contains the review

        Returns:
            str: Extracted review content
        """
        try:
            # First attempt to extract from boxed format
            return pred_context.split(r'\boxed_review{')[-1].split('\n}')[0]
        except Exception:
            # Alternative extraction if the first method fails
            if isinstance(pred_context, dict) and 'output' in pred_context:
                return pred_context['output'].split(r'\boxed_review{')[-1].split('\n}')[0]
            else:
                # Return as is if extraction fails
                return pred_context



def load_json_data(json_path: str) -> List[Dict[str, Any]]:
    """
    Load JSON data from file.
    Handles both list and dict formats.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert dict to list if needed
    if isinstance(data, dict):
        data = list(data.values())
    
    return data


def load_prompt_template(yaml_path: str) -> str:
    """
    Load the rubric generation prompt from YAML file.
    """
    with open(yaml_path, 'r', encoding='utf-8') as f:
        prompts = yaml.safe_load(f)
    
    prompt_template = prompts.get('v2_rubric_generation_prompt', '')
    rubric_template = prompts.get('rubrics', '')
    
    prompt_template = prompt_template.replace('<<rubric_template>>', rubric_template)
    
    return prompt_template


def clean_rubrics_json(json_str: str) -> str:
    """
    Clean JSON string by escaping unescaped double quotes inside string values.
    
    This function handles cases where the model outputs double quotes inside
    string values (especially in description fields) without proper escaping.
    
    The expected format is a JSON array of objects with "title", "description", "weight" fields.
    Strategy: Find each field's value and escape unescaped quotes inside it.
    """
    import re
    
    # First, try to extract JSON array if wrapped in markdown code blocks
    json_match = re.search(r'```json\s*(\[.*?\])\s*```', json_str, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        # Try to find JSON array directly
        json_match = re.search(r'(\[.*?\])', json_str, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
    
    # Process the JSON string character by character to find and fix string values
    # We'll look for patterns like "field": " and then find the matching closing quote
    result = []
    i = 0
    
    while i < len(json_str):
        # Look for field pattern: "field_name": "
        field_match = re.search(r'"(title|description|weight)"\s*:\s*"', json_str[i:])
        if not field_match:
            # No more fields to process, append rest and break
            result.append(json_str[i:])
            break
        
        # Append everything before the match
        match_start = i + field_match.start()
        result.append(json_str[i:match_start])
        
        # Process the field value
        value_start = i + field_match.end()  # Position after opening quote
        
        # Find the closing quote by scanning character by character
        # The closing quote should be followed by comma, closing brace, or closing bracket
        j = value_start
        found_closing = False
        
        while j < len(json_str):
            if json_str[j] == '\\':
                # Skip escaped character (could be \", \\, etc.)
                if j + 1 < len(json_str):
                    j += 2
                    continue
                else:
                    j += 1
                    break
            elif json_str[j] == '"':
                # Found a quote - check if it's the closing quote
                # Look ahead (skip whitespace) to see if followed by comma, brace, or bracket
                k = j + 1
                while k < len(json_str) and json_str[k] in ' \t\n\r':
                    k += 1
                
                if k < len(json_str) and json_str[k] in ',}]':
                    # This is the closing quote!
                    value_content = json_str[value_start:j]
                    closing_part = json_str[j:k+1]  # " followed by , } or ]
                    
                    # Fix unescaped quotes in value_content
                    # Strategy: preserve already-escaped quotes, escape others
                    fixed_content = value_content.replace('\\"', '__TEMP_ESC__')
                    fixed_content = fixed_content.replace('"', '\\"')
                    fixed_content = fixed_content.replace('__TEMP_ESC__', '\\"')
                    
                    # Append the fixed field
                    result.append(json_str[match_start:value_start])  # "field": "
                    result.append(fixed_content)  # fixed value content
                    result.append(closing_part)  # " followed by punctuation
                    
                    i = k + 1
                    found_closing = True
                    break
            j += 1
        
        if not found_closing:
            # Couldn't find proper closing quote, append rest and break
            result.append(json_str[match_start:])
            break
    
    return ''.join(result)


def extract_rubrics_from_response(response: str) -> Optional[List[Dict[str, Any]]]:
    """
    Extract rubrics (JSON array) from LLM response.
    Handles cases where description fields contain unescaped double quotes.
    Returns None if parsing fails (silently, no error messages printed).
    """
    try:
        # First, try using parse_llm_response (handles markdown blocks)
        try:
            parsed = parse_llm_response(response)
            
            # Check if parsed result is a list (array of rubrics)
            if isinstance(parsed, list):
                return parsed
            
            # If parsed result is a dict, check for common keys that might contain the array
            if isinstance(parsed, dict):
                # Check for common keys
                for key in ['rubrics', 'rubric', 'items', 'criteria']:
                    if key in parsed and isinstance(parsed[key], list):
                        return parsed[key]
                
                # If no key found, try to find the first list value
                for value in parsed.values():
                    if isinstance(value, list):
                        return value
        except Exception:
            # parse_llm_response failed, try manual cleaning
            pass
        
        # If parse_llm_response failed, try manual extraction and cleaning
        import re
        
        # Try to find JSON array in response
        json_match = re.search(r'\[.*?\]', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            
            # Try direct parsing first
            try:
                rubrics = json.loads(json_str)
                if isinstance(rubrics, list):
                    return rubrics
            except json.JSONDecodeError:
                # JSON parsing failed, try cleaning
                try:
                    cleaned_json = clean_rubrics_json(json_str)
                    rubrics = json.loads(cleaned_json)
                    if isinstance(rubrics, list):
                        return rubrics
                except Exception:
                    # Last resort: try a more aggressive cleaning approach
                    try:
                        # Replace unescaped quotes in description fields more aggressively
                        # Pattern: "description": "..." where ... may contain quotes
                        def fix_description_quotes(match):
                            prefix = match.group(1)  # "description": "
                            content = match.group(2)  # the content
                            suffix = match.group(3)  # closing quote
                            
                            # Escape all quotes in content, but preserve escaped ones
                            # First, mark escaped quotes temporarily
                            content = content.replace('\\"', '__ESCAPED_QUOTE__')
                            # Escape all remaining quotes
                            content = content.replace('"', '\\"')
                            # Restore escaped quotes
                            content = content.replace('__ESCAPED_QUOTE__', '\\"')
                            
                            return prefix + content + suffix
                        
                        # More specific pattern for description field
                        desc_pattern = r'("description"\s*:\s*")(.*?)("(?:\s*[,}])?)'
                        fixed_json = re.sub(desc_pattern, fix_description_quotes, json_str, flags=re.DOTALL)
                        
                        rubrics = json.loads(fixed_json)
                        if isinstance(rubrics, list):
                            return rubrics
                    except Exception:
                        pass
        
        # If all else fails, return None (silently)
        return None
        
    except Exception:
        # Any unexpected error, return None (silently)
        return None


def generate_rubrics_for_entry(
    entry: Dict[str, Any],
    prompt_template: str,
    llm_service: LLMService,
    max_retries: int = 16
) -> Dict[str, Any]:
    """
    Generate rubrics for a single entry with retry mechanism.
    
    Args:
        entry: Dictionary with 'id', 'pred_fast_mode_baseline', 'paper_context', 'decision'
        prompt_template: Prompt template with <<golden_review>> placeholder
        llm_service: LLMService instance (VLLMService or GPTService)
        max_retries: Maximum number of retries if JSON parsing fails (default: 16)
        
    Returns:
        Dictionary with 'id', 'paper_context', 'decision', 'golden_review', 'rubrics' (list)
    """
    entry_id = entry.get('id', 'unknown')
    golden_review = entry.get('pred_fast_mode_baseline', '')
    paper_context = entry.get('paper_context', '')
    decision = entry.get('decision', '')
    
    # Replace placeholder in prompt template
    prompt = prompt_template.replace('<<golden_review>>', golden_review)
    prompt = prompt.replace('<<paper_context>>', paper_context)
    
    # Convert prompt to messages format (shared/utils services use messages format)
    messages = [{"role": "user", "content": prompt}]
    
    # Retry loop for JSON parsing failures
    last_error = None
    for attempt in range(max_retries):
        try:
            # Generate response from LLM
            response = llm_service.generate(messages=messages)
            
            # Extract rubrics from response
            rubrics_list = extract_rubrics_from_response(response)
            
            # If successful, return the result (silently, no output during retries)
            if rubrics_list is not None and isinstance(rubrics_list, list):
                return {
                    'id': entry_id,
                    'paper_context': paper_context,
                    'decision': decision,
                    'golden_review': golden_review,
                    'rubrics': rubrics_list
                }
            
            # If extraction failed, continue retrying silently
            # Store the error message for the last attempt
            if attempt == max_retries - 1:
                last_error = "Failed to extract valid rubrics from response"
                
        except Exception as e:
            # Store the error (will be overwritten by subsequent attempts until the last one)
            last_error = e
    
    # All retries failed, output warning only once
    if last_error:
        print(f"[WARN] Failed to generate rubrics for entry {entry_id} after {max_retries} attempts: {last_error}")
    
    # All retries failed, return with empty rubrics
    result = {
        'id': entry_id,
        'paper_context': paper_context,
        'decision': decision,
        'golden_review': golden_review,
        'rubrics': []  # Empty list as fallback
    }
    if last_error:
        result['error'] = str(last_error)
    return result


def load_llm_config(config_path: str) -> Dict[str, Any]:
    """
    Load LLM configuration from YAML file.
    
    Args:
        config_path: Path to configs.yaml file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def create_llm_service_from_config(config: Dict[str, Any]) -> LLMService:
    """
    Create LLM service from configuration.
    
    Args:
        config: Configuration dictionary from configs.yaml
        
    Returns:
        LLMService instance (VLLMService or GPTService)
    """
    mode = config.get('mode', 'gpt').lower()
    
    if mode == 'gpt':
        gpt_config = config.get('gpt', {})
        api_key = gpt_config.get('api_key') or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("GPT mode requires api_key in configs.yaml or OPENAI_API_KEY environment variable")
        
        service = GPTService(
            api_key=api_key,
            model_name=gpt_config.get('model_name', 'gpt-4o'),
            base_url=gpt_config.get('base_url'),
            timeout=gpt_config.get('timeout', 300)
        )
        return service
        
    elif mode == 'vllm':
        vllm_config = config.get('vllm', {})
        service = VLLMService(
            base_url=vllm_config.get('base_url', 'http://localhost:8000/v1'),
            api_key=vllm_config.get('api_key', 'dummy-key'),
            model_name=vllm_config.get('model_name'),
            timeout=vllm_config.get('timeout', 300),
            max_concurrent_requests=vllm_config.get('max_concurrent_requests', 64),
            max_retries=vllm_config.get('max_retries', 3),
            retry_delay=vllm_config.get('retry_delay', 1.0),
            retry_backoff=vllm_config.get('retry_backoff', 2.0)
        )
        return service
        
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'gpt' or 'vllm'")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate review-based rubrics using LLMs")
    
    # Input/Output paths
    parser.add_argument("--json_path", type=str, required=True,
                       help="Path to input JSON file with review data")
    parser.add_argument("--output_path", type=str, default=None,
                       help="Path to output JSON file (default: eval_rubrics.json in same dir as input)")
    parser.add_argument("--yaml_path", type=str, default=None,
                       help="Path to prompts.yaml file (default: prompts.yaml in same dir as script)")
    parser.add_argument("--config_path", type=str, default=None,
                       help="Path to configs.yaml file (default: configs.yaml in same dir as script)")
    
    # Multi-threading
    parser.add_argument("--max_workers", type=int, default=None,
                       help="Maximum number of worker threads (default: from MAX_WORKERS env var or 5)")
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # File paths
    json_path = args.json_path
    if not os.path.isabs(json_path):
        json_path = os.path.join(script_dir, json_path)
    
    if args.output_path:
        output_path = args.output_path
        if not os.path.isabs(output_path):
            output_path = os.path.join(script_dir, output_path)
    else:
        # Default: same directory as input JSON, with eval_rubrics.json name
        output_dir = os.path.dirname(json_path)
        output_path = os.path.join(output_dir, 'eval_rubrics.json')
    
    if args.yaml_path:
        yaml_path = args.yaml_path
        if not os.path.isabs(yaml_path):
            yaml_path = os.path.join(script_dir, yaml_path)
    else:
        yaml_path = os.path.join(script_dir, 'prompts.yaml')
    
    if args.config_path:
        config_path = args.config_path
        if not os.path.isabs(config_path):
            config_path = os.path.join(script_dir, config_path)
    else:
        config_path = os.path.join(script_dir, 'configs.yaml')
    
    max_workers = args.max_workers or int(os.getenv("MAX_WORKERS", "5"))
    
    # Check if files exist
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    print(f"Loading JSON data from {json_path}...")
    data = load_json_data(json_path)
    print(f"Loaded {len(data)} entries")
    
    print(f"Loading prompt template from {yaml_path}...")
    prompt_template = load_prompt_template(yaml_path)
    if not prompt_template:
        raise ValueError("Could not find 'v2_rubric_generation_prompt' in YAML file")
    print("Prompt template loaded successfully")
    
    # Load LLM configuration and create service
    print(f"Loading LLM configuration from {config_path}...")
    llm_config = load_llm_config(config_path)
    llm_service = create_llm_service_from_config(llm_config)
    mode = llm_config.get('mode', 'gpt')
    print(f"LLM service initialized (mode: {mode})")
    if hasattr(llm_service, 'model_name'):
        print(f"Using model: {llm_service.model_name}")
    
    # Extract required fields from each entry
    print("Extracting required fields from entries...")
    entries = []
    for item in data:
        
        if 'id' in item and 'pred_fast_mode_baseline' in item:
            entries.append(item)
        else:
            print(f"[WARN] Skipping entry missing required fields: {item.get('id', 'unknown')}")
    
    print(f"Processing {len(entries)} entries with {max_workers} workers...")
    
    # Generate rubrics using concurrent processing
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_entry = {
            executor.submit(
                generate_rubrics_for_entry,
                entry,
                prompt_template,
                llm_service
            ): entry
            for entry in entries
        }
        
        # Process completed tasks with progress bar
        for future in tqdm(as_completed(future_to_entry), total=len(entries), desc="Generating rubrics"):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                entry = future_to_entry[future]
                entry_id = entry.get('id', 'unknown')
                print(f"\n[ERROR] Failed to process entry {entry_id}: {e}")
                # Add error entry with empty rubrics
                results.append({
                    'id': entry_id,
                    'paper_context': entry.get('paper_context', ''),
                    'decision': entry.get('decision', ''),
                    'golden_review': entry.get('pred_fast_mode_baseline', ''),
                    'rubrics': [],
                    'error': str(e)
                })
    
    print(f"\nSuccessfully generated rubrics for {len(results)} entries")
    
    # Save to JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {output_path}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"Total entries processed: {len(results)}")
    
    # Count successful vs failed
    successful = sum(1 for r in results if 'error' not in r and len(r.get('rubrics', [])) > 0)
    failed = len(results) - successful
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    # Check rubrics statistics
    rubric_counts = [len(r.get('rubrics', [])) for r in results if isinstance(r.get('rubrics'), list)]
    
    if rubric_counts:
        print(f"\nRubrics per entry:")
        print(f"  Mean: {sum(rubric_counts) / len(rubric_counts):.2f}")
        print(f"  Min: {min(rubric_counts)}")
        print(f"  Max: {max(rubric_counts)}")


if __name__ == "__main__":
    main()
    
"""
Example usage:
python 1_generate_review_based_rubrics.py \
        --json_path ./examples/input.json \
        --output_path eval_rubrics.json \
        --yaml_path prompts.yaml \
        --config_path configs.yaml \
        --max_workers 5
"""