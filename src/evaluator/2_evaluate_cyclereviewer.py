"""
Unified evaluation script for semantic (LLM-based) and auto_metric (rule-based) evaluation.

This script:
1. Reads eval_rubrics.json (from 1_generate_review_based_rubrics.py) containing rubrics for each paper
2. Reads input JSON file containing model reviews (supports multiple formats)
3. Supports three evaluation modes:
   - semantic: LLM-based rubrics evaluation (from 2_evaluate_direct.py)
   - auto_metric: Rule-based metrics evaluation (from 3_rule_evaluate.py)
   - both: Run both evaluations separately
4. Supports strict mode: normalize scores to discrete scales before computing metrics (--strict_mode)
5. Outputs separate JSON files for results and summaries

Usage:
    # Semantic evaluation only
    python 2_evaluate.py \
        --rubrics_path eval_rubrics.json \
        --reviews_path model_reviews.json \
        --mode semantic \
        --yaml_path prompts.yaml \
        --config_path configs.yaml \
        --semantic_output semantic_results.json \
        --max_workers 5

    # Auto-metric evaluation only
    python 2_evaluate.py \
        --rubrics_path eval_rubrics.json \
        --reviews_path model_reviews.json \
        --mode auto_metric \
        --auto_metric_output auto_metric_results.json

    # Auto-metric evaluation with strict mode (normalize scores to discrete scales)
    python 2_evaluate.py \
        --rubrics_path eval_rubrics.json \
        --reviews_path model_reviews.json \
        --mode auto_metric \
        --auto_metric_output auto_metric_results.json \
        --strict_mode

    # Auto-metric evaluation with manually specified input format (refined)
    python 2_evaluate.py \
        --rubrics_path eval_rubrics.json \
        --reviews_path model_reviews.json \
        --mode auto_metric \
        --auto_metric_output auto_metric_results.json \
        --input_format refined

    # Auto-metric evaluation with manually specified input format (original)
    python 2_evaluate.py \
        --rubrics_path eval_rubrics.json \
        --reviews_path ours.json \
        --mode auto_metric \
        --auto_metric_output auto_metric_results.json \
        --input_format original

    # Both evaluations
    python 2_evaluate.py \
        --rubrics_path eval_rubrics.json \
        --reviews_path model_reviews.json \
        --mode both \
        --yaml_path prompts.yaml \
        --config_path configs.yaml \
        --semantic_output semantic_results.json \
        --auto_metric_output auto_metric_results.json \
        --max_workers 32
"""
from __future__ import annotations

import json
import os
import sys
import argparse
import yaml
import math
import re
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from itertools import combinations
from scipy.stats import spearmanr
from sklearn.metrics import precision_recall_fscore_support

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Import parse_llm_response from local llm_service module
import llm_service as local_llm_service
parse_llm_response = local_llm_service.parse_llm_response

# Import from shared/utils for gpt/vllm support
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from shared.utils.llm_service import LLMService
from shared.utils.vllm_service import VLLMService
from shared.utils.gpt_service import GPTService
sys.path.insert(0, os.path.join(project_root, 'shared', 'utils'))
from json_parser import parse_review_markdown

def convert_cyclereviewer(review_text: str) -> tuple:
    """
    Convert the review text from cyclereviewer format to unified review system format.
    
    The cyclereviewer format has markdown sections like:
    "## Rating\n\n3: reject, not good enough\n\n## Confidence\n\n4: You are confident...\n\n"
    
    Args:
        review_text: Raw review text string (markdown format)
    
    Returns:
        Tuple of (formatted_review_text, meta_review_dict)
    """
    # Use parse_review_markdown to extract scores from markdown sections
    parsed = {}
    try:
        parsed = parse_review_markdown(review_text)
    except Exception:
        pass
    
    # Extract rating - can be from "## Rating\n\n3: reject..." or "## Score: 6: ..."
    rating = parsed.get('rating')
    if rating is None:
        # Try to extract from "## Rating\n\n3: reject..." format
        rating_match = re.search(r'##\s*Rating\s*\n\n\s*(\d+\.?\d*)\s*:', review_text, re.IGNORECASE | re.MULTILINE)
        if rating_match:
            try:
                rating = float(rating_match.group(1))
            except (ValueError, IndexError):
                pass
        
        # Try "## Score: 6: ..." format
        if rating is None:
            score_match = re.search(r'##\s*Score\s*:\s*(\d+\.?\d*)\s*:', review_text, re.IGNORECASE | re.MULTILINE)
            if score_match:
                try:
                    rating = float(score_match.group(1))
                except (ValueError, IndexError):
                    pass
    
    # Extract confidence
    confidence = parsed.get('confidence')
    if confidence is None:
        # Try to extract from "## Confidence\n\n4: You are confident..." format
        confidence_match = re.search(r'##\s*Confidence\s*\n\n\s*(\d+\.?\d*)\s*:', review_text, re.IGNORECASE | re.MULTILINE)
        if confidence_match:
            try:
                confidence = float(confidence_match.group(1))
            except (ValueError, IndexError):
                pass
    
    # Extract decision from rating text (e.g., "3: reject, not good enough")
    decision = None
    if rating is not None:
        # Look for decision in rating section
        rating_section_match = re.search(r'##\s*Rating\s*\n\n(.*?)(?=\n##|$)', review_text, re.IGNORECASE | re.DOTALL)
        if rating_section_match:
            rating_content = rating_section_match.group(1)
            # Extract decision from text like "3: reject, not good enough"
            decision_match = re.search(r':\s*(accept|reject|undecided)', rating_content, re.IGNORECASE)
            if decision_match:
                decision = decision_match.group(1).lower()
        
        # Also try Score section
        if decision is None:
            score_section_match = re.search(r'##\s*Score\s*:\s*\d+\s*:\s*(.*?)(?=\n##|$)', review_text, re.IGNORECASE | re.DOTALL)
            if score_section_match:
                score_content = score_section_match.group(1)
                decision_match = re.search(r'(accept|reject|undecided)', score_content, re.IGNORECASE)
                if decision_match:
                    decision = decision_match.group(1).lower()
    
    # Extract soundness, presentation from parsed data or markdown
    soundness = parsed.get('soundness')
    presentation = parsed.get('presentation')
    contribution = parsed.get('contribution')
    
    # Create meta_review dict
    meta_review = {
        "rating": rating,
        "soundness": soundness,
        "presentation": presentation,
        "contribution": contribution,
        "confidence": confidence,
        "decision": decision,
    }
    
    # Return the review text as-is (it's already in markdown format)
    return review_text, meta_review


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


# ============================================================================
# Semantic Evaluation Functions (from 2_evaluate_direct.py)
# ============================================================================

def load_prompt_template(yaml_path: str) -> str:
    """Load the evaluator prompt from YAML file."""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        prompts = yaml.safe_load(f)
    return prompts.get('v1_evaluator_prompt', '')


def build_evaluation_prompt(
    rubrics: List[Dict[str, Any]],
    paper_content: str,
    review: str,
    prompt_template: str
) -> str:
    """Build the evaluation prompt by replacing placeholders."""
    rubrics_json = json.dumps(rubrics, indent=4, ensure_ascii=False)
    prompt = prompt_template.replace('{rubrics_json}', rubrics_json)
    prompt = prompt.replace('<<paper_content>>', paper_content)
    prompt = prompt.replace('<<review>>', review)
    return prompt


def calculate_weighted_scores(
    raw_scores: Dict[str, Dict[str, Any]], 
    rubrics: List[Dict[str, Any]]
) -> Dict[str, float]:
    """Calculate weighted scores for each rubric."""
    rubric_weights = {r['title']: r['weight'] for r in rubrics}
    weighted_scores = {}
    
    for rubric_title, rubric_data in raw_scores.items():
        if rubric_title not in rubric_weights:
            continue
        
        rubric_score = rubric_data.get('score', 0)
        if isinstance(rubric_score, str):
            try:
                rubric_score = int(rubric_score)
            except ValueError:
                rubric_score = 0
        
        if rubric_score not in [0, 1]:
            rubric_score = 1 if rubric_score > 0 else 0
        
        weight = rubric_weights[rubric_title]
        weighted_scores[rubric_title] = rubric_score * weight
    
    return weighted_scores


def calculate_scores(raw_scores: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    """Calculate scores for each rubric."""
    scores = {}
    for rubric_title, rubric_data in raw_scores.items():
        scores[rubric_title] = rubric_data.get('score', 0)
    return scores


def evaluate_review_semantic(
    entry: Dict[str, Any],
    paper_content: str,
    prompt_template: str,
    llm_service: LLMService
) -> Dict[str, Any]:
    """Evaluate a single review using article-specific rubrics."""
    entry_id = entry.get('id', 'unknown')
    rubrics = entry.get('rubrics', [])
    model_review = entry.get('model_review', '')
    
    if not rubrics:
        return {
            'id': entry_id,
            'raw_scores': {},
            'weighted_scores': {},
            'total_score': 0.0,
            'error': 'No valid rubrics found',
            'raw_response': ''
        }
    
    # Build prompt
    prompt = build_evaluation_prompt(rubrics, paper_content, model_review, prompt_template)
    
    # Call LLM
    try:
        messages = [{"role": "user", "content": prompt}]
        response = llm_service.generate(messages=messages)
        
        # Parse response
        raw_scores = parse_llm_response(response)
        weighted_scores = calculate_scores(raw_scores)
        total_score = sum(weighted_scores.values())
        
        return {
            'id': entry_id,
            'raw_scores': raw_scores,
            'weighted_scores': weighted_scores,
            'total_score': total_score,
            'raw_response': response
        }
    except Exception as e:
        print(f"[ERROR] Error evaluating review {entry_id}: {e}")
        return {
            'id': entry_id,
            'raw_scores': {},
            'weighted_scores': {},
            'total_score': 0.0,
            'error': str(e),
            'raw_response': ''
        }


def calculate_per_rubric_statistics(
    valid_results: List[Dict[str, Any]],
    rubric_titles: List[str]
) -> Dict[str, Dict[str, float]]:
    """Calculate per-rubric statistics from evaluation results."""
    rubric_scores = {title: [] for title in rubric_titles}
    
    for result in valid_results:
        weighted_scores = result.get('weighted_scores', {})
        if not isinstance(weighted_scores, dict):
            continue
        
        for rubric_title in rubric_titles:
            if rubric_title in weighted_scores:
                score = weighted_scores[rubric_title]
                if isinstance(score, str):
                    try:
                        score = float(score)
                    except ValueError:
                        continue
                elif isinstance(score, (int, float)):
                    score = float(score)
                else:
                    continue
                rubric_scores[rubric_title].append(score)
    
    per_rubric_stats = {}
    for rubric_title in rubric_titles:
        scores = rubric_scores[rubric_title]
        if not scores:
            continue
        
        mean_score = sum(scores) / len(scores)
        min_score = min(scores)
        max_score = max(scores)
        count = len(scores)
        
        if rubric_title == "False or Contradictory Claims":
            pass_count = sum(1 for s in scores if s >= 0)
        else:
            pass_count = sum(1 for s in scores if s >= 1)
        pass_rate = pass_count / count if count > 0 else 0.0
        
        per_rubric_stats[rubric_title] = {
            'mean': mean_score,
            'min': min_score,
            'max': max_score,
            'count': count,
            'pass_rate': pass_rate
        }
    
    return per_rubric_stats


# ============================================================================
# Auto-Metric Evaluation Functions (from 3_rule_evaluate.py)
# ============================================================================

def extract_scores_from_review(review_text: str) -> Dict[str, Any]:
    """Extract numeric scores and decision from a review markdown text."""
    if not review_text:
        return {'soundness': None, 'presentation': None, 'rating': None, 'confidence': None, 'decision': None}
    
    try:
        parsed = parse_review_markdown(review_text)
        decision = parsed.get('decision', '')
        if decision:
            decision_lower = decision.lower().strip()
            if 'accept' in decision_lower:
                decision = 'accept'
            elif 'reject' in decision_lower:
                decision = 'reject'
            elif 'undecided' in decision_lower:
                decision = 'undecided'
            else:
                decision = decision_lower
        else:
            decision = None
        
        return {
            'soundness': parsed.get('soundness'),
            'presentation': parsed.get('presentation'),
            'rating': parsed.get('rating'),
            'confidence': parsed.get('confidence'),
            'decision': decision
        }
    except Exception as e:
        print(f"Warning: Failed to parse review text: {e}")
        return {'soundness': None, 'presentation': None, 'rating': None, 'confidence': None, 'decision': None}


def calculate_mse(predicted: float, ground_truth: float) -> Optional[float]:
    """Calculate Mean Squared Error for a single value."""
    if predicted is None or ground_truth is None:
        return None
    return (predicted - ground_truth) ** 2


def calculate_mae(predicted: float, ground_truth: float) -> Optional[float]:
    """Calculate Mean Absolute Error for a single value."""
    if predicted is None or ground_truth is None:
        return None
    return abs(predicted - ground_truth)


def normalize_to_discrete_scale(score: Optional[float], scale_type: str) -> Optional[float]:
    """
    Normalize a float score to the nearest discrete value based on scale type.
    Uses round-half-up tie-breaking (e.g., 3.5 rounds to 4, 1.5 rounds to 2).
    
    Args:
        score: The float score to normalize (can be None)
        scale_type: Either '0-5' for 0-5 scale (discrete: 0,1,2,3,4,5) 
                    or '0-10' for 0-10 scale (discrete: 0,2,4,6,8,10)
    
    Returns:
        Normalized discrete score, or None if input is None
    """
    if score is None:
        return None
    
    try:
        score = float(score)
    except (ValueError, TypeError):
        return None
    
    if scale_type == '0-5':
        # Discrete values: 0, 1, 2, 3, 4, 5
        discrete_values = [0, 1, 2, 3, 4, 5]
        # Clamp to valid range
        score = max(0, min(5, score))
        # Find nearest discrete value, with round-half-up tie-breaking
        # For ties, prefer the higher value
        best_value = None
        best_distance = float('inf')
        for val in discrete_values:
            distance = abs(val - score)
            if distance < best_distance:
                best_distance = distance
                best_value = val
            elif distance == best_distance and val > best_value:
                # Tie-breaking: prefer higher value (round-half-up)
                best_value = val
        return best_value
    elif scale_type == '0-10':
        # Discrete values: 0, 2, 4, 6, 8, 10
        discrete_values = [0, 2, 4, 6, 8, 10]
        # Clamp to valid range
        score = max(0, min(10, score))
        # Find nearest discrete value, with round-half-up tie-breaking
        best_value = None
        best_distance = float('inf')
        for val in discrete_values:
            distance = abs(val - score)
            if distance < best_distance:
                best_distance = distance
                best_value = val
            elif distance == best_distance and val > best_value:
                # Tie-breaking: prefer higher value (round-half-up)
                best_value = val
        return best_value
    else:
        raise ValueError(f"Unknown scale_type: {scale_type}. Must be '0-5' or '0-10'")


def normalize_scores_dict(scores: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
    """
    Normalize all scores in a dictionary to their appropriate discrete scales.
    
    Args:
        scores: Dictionary with keys 'soundness', 'presentation', 'rating', 'confidence'
    
    Returns:
        Dictionary with normalized scores
    """
    normalized = {}
    
    # soundness, presentation, confidence use 0-5 scale
    for key in ['soundness', 'presentation', 'confidence']:
        normalized[key] = normalize_to_discrete_scale(scores.get(key), '0-5')
    
    # rating uses 0-10 scale
    normalized['rating'] = normalize_to_discrete_scale(scores.get('rating'), '0-10')
    
    return normalized


def calculate_score_metrics(
    model_scores: Dict[str, float],
    ground_truth_scores: Dict[str, float],
    normalize: bool = False
) -> Dict[str, Any]:
    """
    Calculate MSE and MAE metrics for each scoring dimension.
    
    Args:
        model_scores: Dictionary with model scores
        ground_truth_scores: Dictionary with ground truth scores
        normalize: If True, normalize scores to discrete scales before computing metrics
    
    Returns:
        Dictionary with MSE, MAE metrics and optionally normalized scores
    """
    dimensions = ['soundness', 'presentation', 'rating', 'confidence']
    
    # Normalize scores to discrete scales if requested
    if normalize:
        model_scores_normalized = normalize_scores_dict(model_scores)
        gt_scores_normalized = normalize_scores_dict(ground_truth_scores)
    else:
        model_scores_normalized = model_scores
        gt_scores_normalized = ground_truth_scores
    
    mse_values = {}
    mae_values = {}
    valid_count = 0
    
    for dim in dimensions:
        # Use normalized scores for metric calculation
        mse = calculate_mse(model_scores_normalized.get(dim), gt_scores_normalized.get(dim))
        mae = calculate_mae(model_scores_normalized.get(dim), gt_scores_normalized.get(dim))
        mse_values[f'{dim}_mse'] = mse
        mae_values[f'{dim}_mae'] = mae
        if mse is not None:
            valid_count += 1
    
    overall_error = sum([v for v in mse_values.values() if v is not None])
    
    result = {
        **mse_values,
        **mae_values,
        'overall_error': overall_error if valid_count > 0 else None,
        'valid_dimensions': valid_count
    }
    
    # Include normalized scores in result for transparency (only if normalize=True)
    if normalize:
        result['model_scores_normalized'] = model_scores_normalized
        result['gt_scores_normalized'] = gt_scores_normalized
    
    return result


def normalize_score_value(value):
    """Normalize score value to float, handling string representations."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        # Try to extract numeric value from string (e.g., "2.75" -> 2.75)
        try:
            import re
            match = re.search(r'(\d+\.?\d*)', value)
            if match:
                return float(match.group(1))
        except:
            pass
    return None


def normalize_decision(decision):
    """Normalize decision string to standard format."""
    if decision is None:
        return None
    decision_lower = str(decision).lower().strip()
    if 'accept' in decision_lower:
        return 'accept'
    elif 'reject' in decision_lower:
        return 'reject'
    elif 'undecided' in decision_lower:
        return 'undecided'
    else:
        return decision_lower


def extract_scores_from_dict(scores_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract scores from a structured dictionary (scores or initial_scores format).
    
    Args:
        scores_dict: Dict containing scores (e.g., {'rating': 5.75, 'soundness': '2.75', ...})
    
    Returns:
        Dict with normalized scores: {'soundness', 'presentation', 'rating', 'confidence', 'decision'}
    """
    if not scores_dict:
        return {
            'soundness': None,
            'presentation': None,
            'rating': None,
            'confidence': None,
            'decision': None
        }
    
    return {
        'soundness': normalize_score_value(scores_dict.get('soundness')),
        'presentation': normalize_score_value(scores_dict.get('presentation')),
        'rating': normalize_score_value(scores_dict.get('rating')),
        'confidence': normalize_score_value(scores_dict.get('confidence')),
        'decision': normalize_decision(scores_dict.get('decision'))
    }


def evaluate_review_auto_metric(entry: Dict[str, Any], use_initial_scores: bool = False, strict_mode: bool = False) -> Dict[str, Any]:
    """
    Evaluate a single entry by extracting scores and calculating metrics.
    
    Args:
        entry: Evaluation entry containing model_review, scores, initial_scores, etc.
        use_initial_scores: If True, use initial_scores instead of refined scores (for refined format)
    
    Returns:
        Dict containing evaluation metrics
    """
    entry_id = entry.get('id', 'unknown')
    model_review = entry.get('model_review', '')
    format_type = entry.get('format', 'unknown')
    
    # Extract scores based on format
    model_scores = {}
    model_decision = None
    
    if format_type == 'refined' and not use_initial_scores:
        # Use refined scores from structured data
        scores_dict = entry.get('scores', {})
        model_data = extract_scores_from_dict(scores_dict)
        model_scores = {
            'soundness': model_data.get('soundness'),
            'presentation': model_data.get('presentation'),
            'rating': model_data.get('rating'),
            'confidence': model_data.get('confidence')
        }
        model_decision = model_data.get('decision')
    elif format_type == 'refined' and use_initial_scores:
        # Use initial scores from structured data
        initial_scores_dict = entry.get('initial_scores', {})
        model_data = extract_scores_from_dict(initial_scores_dict)
        model_scores = {
            'soundness': model_data.get('soundness'),
            'presentation': model_data.get('presentation'),
            'rating': model_data.get('rating'),
            'confidence': model_data.get('confidence')
        }
        model_decision = model_data.get('decision')
    elif format_type == 'original':
        # Use initial scores from structured data
        initial_scores_dict = entry.get('initial_scores', {})
        model_data = extract_scores_from_dict(initial_scores_dict)
        model_scores = {
            'soundness': model_data.get('soundness'),
            'presentation': model_data.get('presentation'),
            'rating': model_data.get('rating'),
            'confidence': model_data.get('confidence')
        }
        model_decision = model_data.get('decision')
        
        # Fallback: If confidence is missing from structured data, try to extract from review text
        # (meta_review may not have confidence field, but review text might)
        if model_scores.get('confidence') is None and model_review:
            try:
                review_data = extract_scores_from_review(model_review)
                if review_data.get('confidence') is not None:
                    model_scores['confidence'] = review_data.get('confidence')
            except Exception:
                pass  # Keep confidence as None if extraction fails
    else:
        # Fallback: extract from markdown review text
        model_data = extract_scores_from_review(model_review)
        model_scores = {
            'soundness': model_data.get('soundness'),
            'presentation': model_data.get('presentation'),
            'rating': model_data.get('rating'),
            'confidence': model_data.get('confidence')
        }
        model_decision = model_data.get('decision')
    
    # Get ground truth scores from golden_review ONLY
    # Ground truth must ONLY come from golden_review, never from model output
    # If extraction fails, leave fields as None (do not use model_review as fallback)
    ground_truth_review = entry.get('golden_review', '')
    ground_truth_scores = {}
    gt_decision = None
    
    if not ground_truth_review:
        print(f"Warning: No golden_review found for entry {entry_id}. Ground truth scores will be empty.")
    else:
        try:
            # Extract scores from golden_review markdown text
            gt_data = extract_scores_from_review(ground_truth_review)
            if not gt_data:
                print(f"Warning: Failed to parse golden_review for entry {entry_id}. Ground truth scores will be empty.")
            else:
                ground_truth_scores = {
                    'soundness': gt_data.get('soundness'),
                    'presentation': gt_data.get('presentation'),
                    'rating': gt_data.get('rating'),
                    'confidence': gt_data.get('confidence')
                }
                gt_decision = normalize_decision(gt_data.get('decision'))
                # Note: If any field is None, it stays None - we do NOT use model_review as fallback
                # Using model output as ground truth would inflate evaluation scores
        except Exception as e:
            print(f"Warning: Failed to extract scores from golden_review for {entry_id}: {e}")
            print(f"  Ground truth scores will be empty. Error: {str(e)}")
    
    # Calculate MSE and MAE metrics (with optional normalization in strict mode)
    score_metrics = calculate_score_metrics(model_scores, ground_truth_scores, normalize=strict_mode)
    
    # Calculate decision accuracy
    decision_match = False
    decision_accuracy = None
    if model_decision is not None and gt_decision is not None:
        model_decision_normalized = normalize_decision(model_decision)
        decision_match = (model_decision_normalized == gt_decision)
        decision_accuracy = 1.0 if decision_match else 0.0
    
    result = {
        'id': entry_id,
        'format': format_type,
        'model_soundness': model_scores.get('soundness'),
        'model_presentation': model_scores.get('presentation'),
        'model_rating': model_scores.get('rating'),
        'model_confidence': model_scores.get('confidence'),
        'model_decision': model_decision,
        'gt_soundness': ground_truth_scores.get('soundness'),
        'gt_presentation': ground_truth_scores.get('presentation'),
        'gt_rating': ground_truth_scores.get('rating'),
        'gt_confidence': ground_truth_scores.get('confidence'),
        'gt_decision': gt_decision,
        'decision_match': decision_match,
        'decision_accuracy': decision_accuracy,
        **score_metrics
    }
    
    # Add prefix to indicate which scores were used
    if format_type == 'refined':
        if use_initial_scores:
            result['score_type'] = 'initial'
        else:
            result['score_type'] = 'refined'
    else:
        result['score_type'] = 'auto'
    
    return result


def calculate_pairwise_accuracies(paper_scores: List[Dict[str, float]]) -> Dict[str, float]:
    """Calculate pairwise accuracy for each metric by comparing rankings."""
    if len(paper_scores) < 2:
        return {}
    
    total_valid_pairs = {'rating': 0, 'soundness': 0, 'presentation': 0, 'confidence': 0}
    correct_pairs = {'rating': 0, 'soundness': 0, 'presentation': 0, 'confidence': 0}
    
    for paper1, paper2 in combinations(paper_scores, 2):
        # Check rating ranking
        if (paper1.get('true_rating') is not None and paper2.get('true_rating') is not None and
            paper1.get('pred_rating') is not None and paper2.get('pred_rating') is not None):
            total_valid_pairs['rating'] += 1
            true_order = paper1['true_rating'] > paper2['true_rating']
            pred_order = paper1['pred_rating'] > paper2['pred_rating']
            if true_order == pred_order:
                correct_pairs['rating'] += 1
        
        # Similar for other dimensions...
        # (abbreviated for space, similar logic for soundness, presentation, confidence)
        for metric in ['soundness', 'presentation', 'confidence']:
            true_key = f'true_{metric}'
            pred_key = f'pred_{metric}'
            if (paper1.get(true_key) is not None and paper2.get(true_key) is not None and
                paper1.get(pred_key) is not None and paper2.get(pred_key) is not None):
                total_valid_pairs[metric] += 1
                true_order = paper1[true_key] > paper2[true_key]
                pred_order = paper1[pred_key] > paper2[pred_key]
                if true_order == pred_order:
                    correct_pairs[metric] += 1
    
    pairwise_accuracies = {
        metric: correct_pairs[metric] / total_valid_pairs[metric] if total_valid_pairs[metric] > 0 else 0.0
        for metric in ['rating', 'soundness', 'presentation', 'confidence']
    }
    
    return pairwise_accuracies


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_rubrics_json(rubrics_path: str) -> Dict[str, Dict[str, Any]]:
    """Load rubrics JSON and create lookup by id."""
    with open(rubrics_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        return {item['id']: item for item in data}
    elif isinstance(data, dict):
        return data
    else:
        raise ValueError(f"Invalid rubrics JSON format: expected list or dict, got {type(data)}")


def load_model_reviews_json(reviews_path: str, format_override: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    Load model reviews JSON and extract reviews by id.
    
    Supports two input formats:
    1. Refined format: Contains 'scores' and 'initial_scores' fields (from refinement pipeline)
    2. Original format: Contains 'model_prediction' with 'meta_review' and 'decision' (like ours.json)
    
    Args:
        reviews_path: Path to JSON file containing model reviews
        format_override: Optional format override ('refined', 'original', or None for auto-detect)
    
    Returns:
        Dict mapping paper_id to dict containing:
        - 'review': review text (markdown)
        - 'scores': refined scores dict (if available)
        - 'initial_scores': initial scores dict (if available)
        - 'format': 'refined' or 'original'
    """
    with open(reviews_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        data = list(data.values())
    
    reviews_dict = {}
    for item in data:
        item_id = None
        review_text = ''
        scores = None
        initial_scores = None
        format_type = None
        
        # Use format override if provided, otherwise auto-detect
        if format_override and format_override != 'auto':
            # Force use specified format
            if format_override == 'refined':
                item_id = item.get('paper_id') or item.get('id')
                if not item_id:
                    continue
                format_type = 'refined'
                review_text = item.get('review_markdown', '') or item.get('review', '')
                scores = item.get('scores', {})
                initial_scores = item.get('initial_scores', {})
            elif format_override == 'original':
                item_id = item.get('id')
                if not item_id:
                    continue
                format_type = 'original'
                model_prediction = item.get('model_prediction', {})
                meta_review = model_prediction.get('meta_review', {})
                review_text = meta_review.get('content', '') or model_prediction.get('raw_text', '')
                initial_scores = {
                    'rating': meta_review.get('rating'),
                    'soundness': meta_review.get('soundness'),
                    'presentation': meta_review.get('presentation'),
                    'contribution': meta_review.get('contribution'),
                    'decision': model_prediction.get('decision'),
                }
            else:
                raise ValueError(f"Unknown format_override: {format_override}. Must be 'refined', 'original', or 'auto'")
        else:
            # Auto-detect format
            if "paper_id" in item:
                # Refined format (from refinement pipeline)
                item_id = item.get('paper_id')
                if not item_id:
                    continue
                
                # Check if this is refined format (has scores and initial_scores)
                if 'scores' in item and 'initial_scores' in item:
                    format_type = 'refined'
                    review_text = item.get('review_markdown', '') or item.get('review', '')
                    scores = item.get('scores', {})
                    initial_scores = item.get('initial_scores', {})
                else:
                    # Standard format with paper_id
                    format_type = 'standard'
                    review_text = item.get('review_markdown', '') or item.get('review', '')
            elif "model_prediction" in item:
                # Original format (like ours.json) or cyclereviewer format
                item_id = item.get('id')
                if not item_id:
                    continue
                
                format_type = 'original'
                model_prediction = item.get('model_prediction', {})
                meta_review = model_prediction.get('meta_review', {})
                
                # Extract review content (prefer meta_review.content, fallback to raw_text)
                review_text = meta_review.get('content', '') or model_prediction.get('raw_text', '')
                
                # Detect cyclereviewer format: has raw_text as markdown string with "## Rating" or "## Score:" patterns
                is_cyclereviewer = False
                if isinstance(review_text, str) and review_text:
                    # Check if it contains cyclereviewer patterns
                    if (re.search(r'##\s*(Rating|Score)\s*:', review_text, re.IGNORECASE) or
                        re.search(r'##\s*Rating\s*\n\n\s*\d+\s*:', review_text, re.IGNORECASE | re.MULTILINE)):
                        is_cyclereviewer = True
                
                # Handle cyclereviewer format
                if is_cyclereviewer:
                    review_text, meta_review = convert_cyclereviewer(review_text)
                
                # Extract initial scores
                # Use meta_review as primary source (from convert_cyclereviewer or original meta_review)
                # Fallback to model_prediction.get('decision') if not in meta_review
                initial_scores = {
                    'rating': meta_review.get('rating'),
                    'soundness': meta_review.get('soundness'),
                    'presentation': meta_review.get('presentation'),
                    'contribution': meta_review.get('contribution'),
                    'confidence': meta_review.get('confidence'),
                    'decision': meta_review.get('decision') or model_prediction.get('decision'),
                }
            else:
                # Legacy format (pred_fast_mode)
                item_id = item.get('id')
                if not item_id:
                    continue
                
                format_type = 'legacy'
                review_dict = item.get('pred_fast_mode', {})
                if isinstance(review_dict, dict):
                    # review_text = review_dict.get('raw_text', '')
                    review_text = review_dict
                else:
                    review_text = str(review_dict)
        
        # Extract review content from the review text field
        try:
            if review_text:
                extracted_review = ReviewProcessor.extract_review_content(review_text)
            else:
                extracted_review = ''
            
            reviews_dict[item_id] = {
                'review': extracted_review,
                'scores': scores,
                'initial_scores': initial_scores,
                'format': format_type
            }
        except Exception as e:
            print(f"[WARN] Failed to extract review for {item_id}: {e}")
            continue
    
    return reviews_dict


def combine_rubrics_and_reviews(
    rubrics_data: Dict[str, Dict[str, Any]],
    reviews_dict: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Combine rubrics and reviews into evaluation entries.
    
    Args:
        rubrics_data: Dict mapping paper_id to rubric entry
        reviews_dict: Dict mapping paper_id to dict containing 'review', 'scores', 'initial_scores', 'format'
    
    Returns:
        List of evaluation entries with model_review, scores, initial_scores, and format info
    """
    combined = []
    missing_reviews = []
    
    for paper_id, rubric_entry in rubrics_data.items():
        review_data = reviews_dict.get(paper_id)
        if not review_data or not review_data.get('review'):
            missing_reviews.append(paper_id)
            continue
        
        entry = {
            'id': paper_id,
            'paper_context': rubric_entry.get('paper_context', ''),
            'decision': rubric_entry.get('decision', ''),
            'golden_review': rubric_entry.get('golden_review', ''),
            'rubrics': rubric_entry.get('rubrics', []),
            'model_review': review_data.get('review', ''),
            'scores': review_data.get('scores'),  # Refined scores (if available)
            'initial_scores': review_data.get('initial_scores'),  # Initial scores (if available)
            'format': review_data.get('format', 'unknown')  # Format type
        }
        combined.append(entry)
    
    if missing_reviews:
        print(f"[WARN] {len(missing_reviews)} papers have no model review, skipping them")
    
    return combined


# ============================================================================
# LLM Service Configuration
# ============================================================================

def load_llm_config(config_path: str) -> Dict[str, Any]:
    """Load LLM configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def create_llm_service_from_config(config: Dict[str, Any]) -> LLMService:
    """Create LLM service from configuration."""
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


# ============================================================================
# Main Evaluation Functions
# ============================================================================

def run_semantic_evaluation(
    evaluation_data: List[Dict[str, Any]],
    prompt_template: str,
    llm_service: LLMService,
    max_workers: int
) -> tuple:
    """Run semantic evaluation and return results and summary."""
    print(f"\n{'='*80}")
    print("RUNNING SEMANTIC EVALUATION")
    print(f"{'='*80}")
    print(f"Evaluating {len(evaluation_data)} reviews using {max_workers} workers...")
    
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_entry = {
            executor.submit(
                evaluate_review_semantic,
                entry,
                entry['paper_context'],
                prompt_template,
                llm_service
            ): entry
            for entry in evaluation_data
        }
        
        for future in tqdm(as_completed(future_to_entry), total=len(evaluation_data), desc="Semantic evaluation"):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                entry = future_to_entry[future]
                print(f"\n[ERROR] Failed to process entry {entry.get('id', 'unknown')}: {e}")
                results.append({
                    'id': entry.get('id', 'unknown'),
                    'raw_scores': {},
                    'weighted_scores': {},
                    'total_score': 0.0,
                    'error': str(e),
                    'raw_response': ''
                })
    
    # Calculate statistics
    valid_results = [r for r in results if 'error' not in r and r.get('weighted_scores')]
    review_scores = [r.get('total_score', 0.0) for r in valid_results]
    
    summary = {
        'total_entries': len(results),
        'valid_entries': len(valid_results),
        'failed_entries': len(results) - len(valid_results)
    }
    
    if review_scores:
        summary['overall_score'] = {
            'mean': sum(review_scores) / len(review_scores),
            'min': min(review_scores),
            'max': max(review_scores)
        }
    
    # Calculate per-rubric statistics (extract rubric titles from first entry)
    if evaluation_data and evaluation_data[0].get('rubrics'):
        rubric_titles = [r['title'] for r in evaluation_data[0]['rubrics']]
        per_rubric_stats = calculate_per_rubric_statistics(valid_results, rubric_titles)
        summary['per_rubric_statistics'] = per_rubric_stats
    
    return results, summary


def run_auto_metric_evaluation(
    evaluation_data: List[Dict[str, Any]],
    strict_mode: bool = False
) -> tuple:
    """
    Run auto-metric evaluation and return results and summary.
    
    For refined format (has scores and initial_scores), evaluates both:
    - Refined scores evaluation
    - Initial scores evaluation
    
    For original format (only initial_scores), evaluates:
    - Initial scores evaluation only
    
    Returns:
        Tuple of (results_list, summary_dict)
        - results_list: List of evaluation results (may contain both refined and initial results for refined format)
        - summary_dict: Summary statistics
    """
    print(f"\n{'='*80}")
    print("RUNNING AUTO-METRIC EVALUATION")
    print(f"{'='*80}")
    print(f"Evaluating {len(evaluation_data)} entries...")
    
    # Detect format types
    refined_format_count = sum(1 for e in evaluation_data if e.get('format') == 'refined')
    original_format_count = sum(1 for e in evaluation_data if e.get('format') == 'original')
    
    if refined_format_count > 0:
        print(f"Detected {refined_format_count} entries in refined format (will evaluate both refined and initial scores)")
    if original_format_count > 0:
        print(f"Detected {original_format_count} entries in original format (will evaluate initial scores only)")
    
    results = []
    for entry in tqdm(evaluation_data, desc="Auto-metric evaluation"):
        format_type = entry.get('format', 'unknown')
        
        if format_type == 'refined':
            # Evaluate both refined scores and initial scores
            try:
                entry_id = entry.get('id', 'unknown')
                
                # Evaluate refined scores
                refined_result = evaluate_review_auto_metric(entry, use_initial_scores=False, strict_mode=strict_mode)
                refined_result['paper_id'] = entry_id  # Keep original paper_id
                refined_result['id'] = f"{entry_id}_refined"
                results.append(refined_result)
                
                # Evaluate initial scores
                initial_result = evaluate_review_auto_metric(entry, use_initial_scores=True, strict_mode=strict_mode)
                initial_result['paper_id'] = entry_id  # Keep original paper_id
                initial_result['id'] = f"{entry_id}_initial"
                results.append(initial_result)
            except Exception as e:
                print(f"Error evaluating entry {entry.get('id', 'unknown')}: {e}")
                results.append({
                    'id': entry.get('id', 'unknown'),
                    'error': str(e)
                })
        else:
            # Evaluate initial scores only (or extract from markdown)
            try:
                result = evaluate_review_auto_metric(entry, use_initial_scores=False, strict_mode=strict_mode)
                results.append(result)
            except Exception as e:
                print(f"Error evaluating entry {entry.get('id', 'unknown')}: {e}")
                results.append({
                    'id': entry.get('id', 'unknown'),
                    'error': str(e)
                })
    
    # Calculate statistics
    valid_results = [r for r in results if 'error' not in r]
    mse_results = [r for r in valid_results if r.get('overall_error') is not None]
    
    # Separate refined and initial results for refined format
    refined_results = [r for r in valid_results if r.get('score_type') == 'refined']
    initial_results = [r for r in valid_results if r.get('score_type') == 'initial']
    auto_results = [r for r in valid_results if r.get('score_type') == 'auto' or r.get('score_type') is None]
    
    summary = {
        'total_entries': len(results),
        'valid_entries': len(valid_results),
        'mse_entries': len(mse_results),
        'refined_results_count': len(refined_results),
        'initial_results_count': len(initial_results),
        'auto_results_count': len(auto_results)
    }
    
    # Calculate MSE/MAE statistics
    # For refined format, only use refined results for overall statistics (avoid double counting)
    # For other formats, use all results
    if refined_format_count > 0:
        # Refined format: use only refined results for overall statistics
        stats_results = [r for r in refined_results if r.get('overall_error') is not None]
    else:
        # Original/other formats: use all results
        stats_results = mse_results
    
    if stats_results:
        dimensions = ['soundness', 'presentation', 'confidence', 'rating']
        mse_stats = {}
        mae_stats = {}
        
        for dim in dimensions:
            mse_list = [r.get(f'{dim}_mse') for r in stats_results if r.get(f'{dim}_mse') is not None]
            mae_list = [r.get(f'{dim}_mae') for r in stats_results if r.get(f'{dim}_mae') is not None]
            
            mse_clean = [x for x in mse_list if x is not None and not (isinstance(x, float) and math.isnan(x))]
            mae_clean = [x for x in mae_list if x is not None and not (isinstance(x, float) and math.isnan(x))]
            
            if mse_clean:
                mse_stats[dim] = {
                    'mean': sum(mse_clean) / len(mse_clean),
                    'count': len(mse_clean)
                }
            if mae_clean:
                mae_stats[dim] = {
                    'mean': sum(mae_clean) / len(mae_clean),
                    'count': len(mae_clean)
                }
        
        overall_errors = [r.get('overall_error') for r in stats_results if r.get('overall_error') is not None]
        overall_clean = [x for x in overall_errors if x is not None and not (isinstance(x, float) and math.isnan(x))]
        
        if overall_clean:
            summary['overall_error'] = {
                'mean': sum(overall_clean) / len(overall_clean),
                'count': len(overall_clean)
            }
        
        summary['mse_statistics'] = mse_stats
        summary['mae_statistics'] = mae_stats
        
        # Calculate separate statistics for refined and initial results
        if refined_results:
            refined_mse_results = [r for r in refined_results if r.get('overall_error') is not None]
            if refined_mse_results:
                refined_mse_stats = {}
                refined_mae_stats = {}
                for dim in dimensions:
                    mse_list = [r.get(f'{dim}_mse') for r in refined_mse_results if r.get(f'{dim}_mse') is not None]
                    mae_list = [r.get(f'{dim}_mae') for r in refined_mse_results if r.get(f'{dim}_mae') is not None]
                    mse_clean = [x for x in mse_list if x is not None and not (isinstance(x, float) and math.isnan(x))]
                    mae_clean = [x for x in mae_list if x is not None and not (isinstance(x, float) and math.isnan(x))]
                    if mse_clean:
                        refined_mse_stats[dim] = {'mean': sum(mse_clean) / len(mse_clean), 'count': len(mse_clean)}
                    if mae_clean:
                        refined_mae_stats[dim] = {'mean': sum(mae_clean) / len(mae_clean), 'count': len(mae_clean)}
                summary['refined_mse_statistics'] = refined_mse_stats
                summary['refined_mae_statistics'] = refined_mae_stats
        
        if initial_results:
            initial_mse_results = [r for r in initial_results if r.get('overall_error') is not None]
            if initial_mse_results:
                initial_mse_stats = {}
                initial_mae_stats = {}
                for dim in dimensions:
                    mse_list = [r.get(f'{dim}_mse') for r in initial_mse_results if r.get(f'{dim}_mse') is not None]
                    mae_list = [r.get(f'{dim}_mae') for r in initial_mse_results if r.get(f'{dim}_mae') is not None]
                    mse_clean = [x for x in mse_list if x is not None and not (isinstance(x, float) and math.isnan(x))]
                    mae_clean = [x for x in mae_list if x is not None and not (isinstance(x, float) and math.isnan(x))]
                    if mse_clean:
                        initial_mse_stats[dim] = {'mean': sum(mse_clean) / len(mse_clean), 'count': len(mse_clean)}
                    if mae_clean:
                        initial_mae_stats[dim] = {'mean': sum(mae_clean) / len(mae_clean), 'count': len(mae_clean)}
                summary['initial_mse_statistics'] = initial_mse_stats
                summary['initial_mae_statistics'] = initial_mae_stats
    
    # Calculate Spearman correlations
    def filter_valid_pairs(true_list, pred_list):
        filtered_true = []
        filtered_pred = []
        for t, p in zip(true_list, pred_list):
            if (t is not None and p is not None and 
                not (isinstance(t, float) and math.isnan(t)) and
                not (isinstance(p, float) and math.isnan(p))):
                filtered_true.append(t)
                filtered_pred.append(p)
        return filtered_true, filtered_pred
    
    # Calculate Spearman correlations
    # For refined format, calculate separately for refined and initial, and use refined for overall
    # For other formats, use all results
    if refined_format_count > 0:
        # Calculate refined spearman correlations
        refined_spearman_stats = {}
        dimensions = ['soundness', 'presentation', 'confidence', 'rating']
        for dim in dimensions:
            true_values = [r.get(f'gt_{dim}') for r in refined_results]
            pred_values = [r.get(f'model_{dim}') for r in refined_results]
            true_clean, pred_clean = filter_valid_pairs(true_values, pred_values)
            
            if len(true_clean) >= 2 and len(pred_clean) >= 2:
                try:
                    corr, _ = spearmanr(true_clean, pred_clean)
                    if not math.isnan(corr):
                        refined_spearman_stats[dim] = {
                            'correlation': corr,
                            'count': len(true_clean)
                        }
                except Exception:
                    pass
        
        # Calculate initial spearman correlations
        initial_spearman_stats = {}
        for dim in dimensions:
            true_values = [r.get(f'gt_{dim}') for r in initial_results]
            pred_values = [r.get(f'model_{dim}') for r in initial_results]
            true_clean, pred_clean = filter_valid_pairs(true_values, pred_values)
            
            if len(true_clean) >= 2 and len(pred_clean) >= 2:
                try:
                    corr, _ = spearmanr(true_clean, pred_clean)
                    if not math.isnan(corr):
                        initial_spearman_stats[dim] = {
                            'correlation': corr,
                            'count': len(true_clean)
                        }
                except Exception:
                    pass
        
        # Use refined for overall statistics (avoid double counting)
        summary['spearman_correlations'] = refined_spearman_stats
        summary['refined_spearman_correlations'] = refined_spearman_stats
        summary['initial_spearman_correlations'] = initial_spearman_stats
    else:
        # Original/other formats: use all results
        correlation_results = valid_results
        spearman_stats = {}
        dimensions = ['soundness', 'presentation', 'confidence', 'rating']
        for dim in dimensions:
            true_values = [r.get(f'gt_{dim}') for r in correlation_results]
            pred_values = [r.get(f'model_{dim}') for r in correlation_results]
            true_clean, pred_clean = filter_valid_pairs(true_values, pred_values)
            
            if len(true_clean) >= 2 and len(pred_clean) >= 2:
                try:
                    corr, _ = spearmanr(true_clean, pred_clean)
                    if not math.isnan(corr):
                        spearman_stats[dim] = {
                            'correlation': corr,
                            'count': len(true_clean)
                        }
                except Exception:
                    pass
        
        summary['spearman_correlations'] = spearman_stats
    
    # Calculate Decision metrics
    # For refined format, calculate separately for refined and initial, and use refined for overall
    # For other formats, use all results
    if refined_format_count > 0:
        # Calculate refined decision metrics
        refined_decision_results = [r for r in refined_results if r.get('gt_decision') is not None and r.get('model_decision') is not None]
        if refined_decision_results:
            true_decisions = []
            pred_decisions = []
            decision_acc = []
            
            for r in refined_decision_results:
                gt_decision = str(r.get('gt_decision', '')).lower().strip()
                pred_decision = str(r.get('model_decision', '')).lower().strip()
                
                if 'accept' in pred_decision:
                    pred_binary = 1
                else:
                    pred_binary = 0
                
                if 'accept' in gt_decision:
                    gt_binary = 1
                else:
                    gt_binary = 0
                
                true_decisions.append(gt_binary)
                pred_decisions.append(pred_binary)
                
                if pred_decision == gt_decision or ('accept' in pred_decision and 'accept' in gt_decision) or ('reject' in pred_decision and 'reject' in gt_decision):
                    decision_acc.append(1.0)
                else:
                    decision_acc.append(0.0)
            
            if decision_acc:
                decision_accuracy = sum(decision_acc) / len(decision_acc)
                try:
                    _, _, f1_score, _ = precision_recall_fscore_support(true_decisions, pred_decisions, average='macro')
                    refined_decision_metrics = {
                        'accuracy': decision_accuracy,
                        'f1_macro': f1_score,
                        'count': len(decision_acc)
                    }
                except Exception:
                    refined_decision_metrics = {
                        'accuracy': decision_accuracy,
                        'count': len(decision_acc)
                    }
                summary['refined_decision_metrics'] = refined_decision_metrics
                summary['decision_metrics'] = refined_decision_metrics  # Use refined for overall
        
        # Calculate initial decision metrics
        initial_decision_results = [r for r in initial_results if r.get('gt_decision') is not None and r.get('model_decision') is not None]
        if initial_decision_results:
            true_decisions = []
            pred_decisions = []
            decision_acc = []
            
            for r in initial_decision_results:
                gt_decision = str(r.get('gt_decision', '')).lower().strip()
                pred_decision = str(r.get('model_decision', '')).lower().strip()
                
                if 'accept' in pred_decision:
                    pred_binary = 1
                else:
                    pred_binary = 0
                
                if 'accept' in gt_decision:
                    gt_binary = 1
                else:
                    gt_binary = 0
                
                true_decisions.append(gt_binary)
                pred_decisions.append(pred_binary)
                
                if pred_decision == gt_decision or ('accept' in pred_decision and 'accept' in gt_decision) or ('reject' in pred_decision and 'reject' in gt_decision):
                    decision_acc.append(1.0)
                else:
                    decision_acc.append(0.0)
            
            if decision_acc:
                decision_accuracy = sum(decision_acc) / len(decision_acc)
                try:
                    _, _, f1_score, _ = precision_recall_fscore_support(true_decisions, pred_decisions, average='macro')
                    initial_decision_metrics = {
                        'accuracy': decision_accuracy,
                        'f1_macro': f1_score,
                        'count': len(decision_acc)
                    }
                except Exception:
                    initial_decision_metrics = {
                        'accuracy': decision_accuracy,
                        'count': len(decision_acc)
                    }
                summary['initial_decision_metrics'] = initial_decision_metrics
    else:
        # Original/other formats: use all results
        decision_results = [r for r in valid_results if r.get('gt_decision') is not None and r.get('model_decision') is not None]
        if decision_results:
            true_decisions = []
            pred_decisions = []
            decision_acc = []
            
            for r in decision_results:
                gt_decision = str(r.get('gt_decision', '')).lower().strip()
                pred_decision = str(r.get('model_decision', '')).lower().strip()
                
                if 'accept' in pred_decision:
                    pred_binary = 1
                else:
                    pred_binary = 0
                
                if 'accept' in gt_decision:
                    gt_binary = 1
                else:
                    gt_binary = 0
                
                true_decisions.append(gt_binary)
                pred_decisions.append(pred_binary)
                
                if pred_decision == gt_decision or ('accept' in pred_decision and 'accept' in gt_decision) or ('reject' in pred_decision and 'reject' in gt_decision):
                    decision_acc.append(1.0)
                else:
                    decision_acc.append(0.0)
            
            if decision_acc:
                decision_accuracy = sum(decision_acc) / len(decision_acc)
                try:
                    _, _, f1_score, _ = precision_recall_fscore_support(true_decisions, pred_decisions, average='macro')
                    summary['decision_metrics'] = {
                        'accuracy': decision_accuracy,
                        'f1_macro': f1_score,
                        'count': len(decision_acc)
                    }
                except Exception:
                    summary['decision_metrics'] = {
                        'accuracy': decision_accuracy,
                        'count': len(decision_acc)
                    }
    
    # Calculate Pairwise comparison
    # For refined format, only use refined results (avoid double counting)
    # For other formats, use all results
    if refined_format_count > 0:
        pairwise_results = refined_results
    else:
        pairwise_results = valid_results
    
    paper_scores = []
    for r in pairwise_results:
        if (r.get('gt_rating') is not None and r.get('model_rating') is not None) or \
           (r.get('gt_soundness') is not None and r.get('model_soundness') is not None):
            paper_scores.append({
                'true_rating': r.get('gt_rating'),
                'pred_rating': r.get('model_rating'),
                'true_soundness': r.get('gt_soundness'),
                'pred_soundness': r.get('model_soundness'),
                'true_presentation': r.get('gt_presentation'),
                'pred_presentation': r.get('model_presentation'),
                'true_confidence': r.get('gt_confidence'),
                'pred_confidence': r.get('model_confidence')
            })
    
    if len(paper_scores) >= 2:
        pairwise_accuracies = calculate_pairwise_accuracies(paper_scores)
        summary['pairwise_accuracies'] = pairwise_accuracies
    
    return results, summary


# ============================================================================
# Main Function
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Unified evaluation script for semantic and auto-metric evaluation")
    
    # Input paths
    parser.add_argument("--rubrics_path", type=str, required=True,
                       help="Path to eval_rubrics.json file (from 1_generate_review_based_rubrics.py)")
    parser.add_argument("--reviews_path", type=str, required=True,
                       help="Path to JSON file with model reviews (contains pred_fast_mode)")
    
    # Evaluation mode
    parser.add_argument("--mode", type=str, choices=["semantic", "auto_metric", "both"], default="both",
                       help="Evaluation mode: semantic (LLM-based), auto_metric (rule-based), or both")
    
    # Output paths
    parser.add_argument("--semantic_output", type=str, default=None,
                       help="Path to output JSON file for semantic evaluation results (required if mode is semantic or both)")
    parser.add_argument("--auto_metric_output", type=str, default=None,
                       help="Path to output JSON file for auto-metric evaluation results (required if mode is auto_metric or both)")
    
    # Semantic evaluation settings
    parser.add_argument("--yaml_path", type=str, default=None,
                       help="Path to prompts.yaml file (required for semantic evaluation)")
    parser.add_argument("--config_path", type=str, default=None,
                       help="Path to configs.yaml file (required for semantic evaluation)")
    
    # Multi-threading
    parser.add_argument("--max_workers", type=int, default=None,
                       help="Maximum number of worker threads for semantic evaluation (default: 5)")
    
    # Strict mode (normalize scores to discrete scales)
    parser.add_argument("--strict_mode", action="store_true", default=False,
                       help="Enable strict mode: normalize scores to discrete scales before computing metrics (default: False)")
    
    # Input format override
    parser.add_argument("--input_format", type=str, choices=['auto', 'refined', 'original'], default='auto',
                       help="Manually specify input JSON format: 'refined' (has scores and initial_scores), 'original' (has model_prediction), or 'auto' for auto-detection (default: 'auto')")
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Resolve paths
    rubrics_path = args.rubrics_path
    if not os.path.isabs(rubrics_path):
        rubrics_path = os.path.join(script_dir, rubrics_path)
    
    reviews_path = args.reviews_path
    if not os.path.isabs(reviews_path):
        reviews_path = os.path.join(script_dir, reviews_path)
    
    max_workers = args.max_workers or int(os.getenv("MAX_WORKERS", "5"))
    
    # Validate mode and output paths
    if args.mode in ["semantic", "both"]:
        if not args.semantic_output:
            raise ValueError("--semantic_output is required when mode is 'semantic' or 'both'")
        if not args.yaml_path:
            raise ValueError("--yaml_path is required for semantic evaluation")
        if not args.config_path:
            raise ValueError("--config_path is required for semantic evaluation")
    
    if args.mode in ["auto_metric", "both"]:
        if not args.auto_metric_output:
            raise ValueError("--auto_metric_output is required when mode is 'auto_metric' or 'both'")
    
    # Check if files exist
    if not os.path.exists(rubrics_path):
        raise FileNotFoundError(f"Rubrics file not found: {rubrics_path}")
    if not os.path.exists(reviews_path):
        raise FileNotFoundError(f"Reviews file not found: {reviews_path}")
    
    # Load data
    print(f"Loading rubrics from {rubrics_path}...")
    rubrics_data = load_rubrics_json(rubrics_path)
    print(f"Loaded {len(rubrics_data)} rubrics entries")
    
    print(f"Loading model reviews from {reviews_path}...")
    if args.input_format != 'auto':
        print(f"Using manually specified format: {args.input_format}")
    else:
        print("Auto-detecting input format...")
    reviews_dict = load_model_reviews_json(reviews_path, format_override=args.input_format if args.input_format != 'auto' else None)
    print(f"Loaded {len(reviews_dict)} model reviews")
    
    # Combine rubrics and reviews
    print("Combining rubrics and reviews...")
    evaluation_data = combine_rubrics_and_reviews(rubrics_data, reviews_dict)
    print(f"Prepared {len(evaluation_data)} entries for evaluation")
    
    # Run evaluations based on mode
    if args.mode in ["semantic", "both"]:
        # Resolve semantic evaluation paths
        yaml_path = args.yaml_path
        if not os.path.isabs(yaml_path):
            yaml_path = os.path.join(script_dir, yaml_path)
        
        config_path = args.config_path
        if not os.path.isabs(config_path):
            config_path = os.path.join(script_dir, config_path)
        
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"YAML file not found: {yaml_path}")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # Load prompt template
        print(f"Loading prompt template from {yaml_path}...")
        prompt_template = load_prompt_template(yaml_path)
        if not prompt_template:
            raise ValueError("Could not find 'v1_evaluator_prompt' in YAML file")
        
        # Initialize LLM service
        print(f"Loading LLM configuration from {config_path}...")
        llm_config = load_llm_config(config_path)
        llm_service = create_llm_service_from_config(llm_config)
        mode = llm_config.get('mode', 'gpt')
        print(f"LLM service initialized (mode: {mode})")
        if hasattr(llm_service, 'model_name'):
            print(f"Using model: {llm_service.model_name}")
        
        # Run semantic evaluation
        semantic_results, semantic_summary = run_semantic_evaluation(
            evaluation_data, prompt_template, llm_service, max_workers
        )
        
        # Save semantic results
        semantic_output = args.semantic_output
        if not os.path.isabs(semantic_output):
            semantic_output = os.path.join(script_dir, semantic_output)
        
        output_dir = os.path.dirname(semantic_output)
        os.makedirs(output_dir, exist_ok=True)
        
        with open(semantic_output, 'w', encoding='utf-8') as f:
            json.dump(semantic_results, f, ensure_ascii=False, indent=2)
        print(f"\nSemantic evaluation results saved to {semantic_output}")
        
        # Save semantic summary
        semantic_summary_path = semantic_output.replace('.json', '_summary.json')
        with open(semantic_summary_path, 'w', encoding='utf-8') as f:
            json.dump(semantic_summary, f, ensure_ascii=False, indent=2)
        print(f"Semantic evaluation summary saved to {semantic_summary_path}")
        
        # Print semantic summary
        print("\n" + "="*80)
        print("SEMANTIC EVALUATION SUMMARY")
        print("="*80)
        print(f"Total entries: {semantic_summary['total_entries']}")
        print(f"Valid entries: {semantic_summary['valid_entries']}")
        print(f"Failed entries: {semantic_summary['failed_entries']}")
        if 'overall_score' in semantic_summary:
            score = semantic_summary['overall_score']
            print(f"\nOverall Score:")
            print(f"  Mean: {score['mean']:.2f}")
            print(f"  Min: {score['min']:.2f}")
            print(f"  Max: {score['max']:.2f}")
    
    if args.mode in ["auto_metric", "both"]:
        # Run auto-metric evaluation
        auto_metric_results, auto_metric_summary = run_auto_metric_evaluation(
            evaluation_data, 
            strict_mode=args.strict_mode
        )
        
        # Save auto-metric results
        auto_metric_output = args.auto_metric_output
        if not os.path.isabs(auto_metric_output):
            auto_metric_output = os.path.join(script_dir, auto_metric_output)
        
        output_dir = os.path.dirname(auto_metric_output)
        os.makedirs(output_dir, exist_ok=True)
        
        with open(auto_metric_output, 'w', encoding='utf-8') as f:
            json.dump(auto_metric_results, f, ensure_ascii=False, indent=2)
        print(f"\nAuto-metric evaluation results saved to {auto_metric_output}")
        
        # Save auto-metric summary
        auto_metric_summary_path = auto_metric_output.replace('.json', '_summary.json')
        with open(auto_metric_summary_path, 'w', encoding='utf-8') as f:
            json.dump(auto_metric_summary, f, ensure_ascii=False, indent=2)
        print(f"Auto-metric evaluation summary saved to {auto_metric_summary_path}")
        
        # Print auto-metric summary
        print("\n" + "="*80)
        print("AUTO-METRIC EVALUATION SUMMARY")
        print("="*80)
        print(f"Total entries: {auto_metric_summary['total_entries']}")
        print(f"Valid entries: {auto_metric_summary['valid_entries']}")
        print(f"MSE entries: {auto_metric_summary['mse_entries']}")
        
        if 'mse_statistics' in auto_metric_summary:
            print("\nMSE Statistics:")
            for dim, stats in auto_metric_summary['mse_statistics'].items():
                print(f"  {dim.capitalize()}: Mean={stats['mean']:.4f}, Count={stats['count']}")
        
        if 'mae_statistics' in auto_metric_summary:
            print("\nMAE Statistics:")
            for dim, stats in auto_metric_summary['mae_statistics'].items():
                print(f"  {dim.capitalize()}: Mean={stats['mean']:.4f}, Count={stats['count']}")
        
        # Print refined and initial statistics if available
        if 'refined_mse_statistics' in auto_metric_summary:
            print("\nRefined Scores - MSE Statistics:")
            for dim, stats in auto_metric_summary['refined_mse_statistics'].items():
                print(f"  {dim.capitalize()}: Mean={stats['mean']:.4f}, Count={stats['count']}")
        
        if 'refined_mae_statistics' in auto_metric_summary:
            print("\nRefined Scores - MAE Statistics:")
            for dim, stats in auto_metric_summary['refined_mae_statistics'].items():
                print(f"  {dim.capitalize()}: Mean={stats['mean']:.4f}, Count={stats['count']}")
        
        if 'initial_mse_statistics' in auto_metric_summary:
            print("\nInitial Scores - MSE Statistics:")
            for dim, stats in auto_metric_summary['initial_mse_statistics'].items():
                print(f"  {dim.capitalize()}: Mean={stats['mean']:.4f}, Count={stats['count']}")
        
        if 'initial_mae_statistics' in auto_metric_summary:
            print("\nInitial Scores - MAE Statistics:")
            for dim, stats in auto_metric_summary['initial_mae_statistics'].items():
                print(f"  {dim.capitalize()}: Mean={stats['mean']:.4f}, Count={stats['count']}")
        
        if 'spearman_correlations' in auto_metric_summary:
            print("\nSpearman Correlations:")
            for dim, stats in auto_metric_summary['spearman_correlations'].items():
                print(f"  {dim.capitalize()}: {stats['correlation']:.4f} (n={stats['count']})")
        
        # Print refined and initial spearman correlations if available
        if 'refined_spearman_correlations' in auto_metric_summary:
            print("\nRefined Scores - Spearman Correlations:")
            for dim, stats in auto_metric_summary['refined_spearman_correlations'].items():
                print(f"  {dim.capitalize()}: {stats['correlation']:.4f} (n={stats['count']})")
        
        if 'initial_spearman_correlations' in auto_metric_summary:
            print("\nInitial Scores - Spearman Correlations:")
            for dim, stats in auto_metric_summary['initial_spearman_correlations'].items():
                print(f"  {dim.capitalize()}: {stats['correlation']:.4f} (n={stats['count']})")
        
        if 'decision_metrics' in auto_metric_summary:
            dm = auto_metric_summary['decision_metrics']
            print(f"\nDecision Metrics:")
            print(f"  Accuracy: {dm['accuracy']:.4f} (n={dm['count']})")
            if 'f1_macro' in dm:
                print(f"  F1 (macro): {dm['f1_macro']:.4f}")
        
        # Print refined and initial decision metrics if available
        if 'refined_decision_metrics' in auto_metric_summary:
            print("\nRefined Scores - Decision Metrics:")
            rdm = auto_metric_summary['refined_decision_metrics']
            print(f"  Accuracy: {rdm['accuracy']:.4f} (n={rdm['count']})")
            if 'f1_macro' in rdm:
                print(f"  F1 (macro): {rdm['f1_macro']:.4f}")
        
        if 'initial_decision_metrics' in auto_metric_summary:
            print("\nInitial Scores - Decision Metrics:")
            idm = auto_metric_summary['initial_decision_metrics']
            print(f"  Accuracy: {idm['accuracy']:.4f} (n={idm['count']})")
            if 'f1_macro' in idm:
                print(f"  F1 (macro): {idm['f1_macro']:.4f}")
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
    
