"""
Reviewer Agent Module

Tool-augmented review generation system with two-stage architecture.
"""
import sys
from pathlib import Path

# Add project root to path for shared utils
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import shared utils
# Use direct imports to avoid lazy import issues
from shared.utils.llm_service import LLMService, ChatMessage
from shared.utils.prompt_loader import get_prompt_loader
from shared.utils.llm_service_factory import get_llm_service_factory
from shared.utils.review_logger import ReviewLogger

# Import main components
from .paper_reviewer import PaperReviewer
from .review_refiner import ReviewRefiner
from .related_work_searcher import RelatedWorkSearcher
from .paper_results_analyzer import PaperResultsAnalyzer    
from .paper_insight_miner import PaperInsightMiner
from .main_pipeline import (
    create_reviewer_pipeline,
    review_paper_from_dict,
    create_review_pipeline_with_refiner,
    review_paper_with_refiner,
)

__all__ = [
    "PaperReviewer",
    "ReviewRefiner",
    "RelatedWorkSearcher",
    "PaperResultsAnalyzer",
    "PaperInsightMiner",
    "create_reviewer_pipeline",
    "review_paper_from_dict",
    "create_review_pipeline_with_refiner",
    "review_paper_with_refiner",
]
