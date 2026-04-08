"""
Command-line interface for the paper reviewer agent
"""
import argparse
import json
import sys
import os
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Suppress httpx INFO logs (HTTP Request messages)
logging.getLogger("httpx").setLevel(logging.WARNING)

from src.reviewer_agent.main_pipeline import create_reviewer_pipeline, review_paper_from_dict


def load_paper_from_file(file_path: str) -> dict:
    """Load paper data from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Paper Reviewer Agent - Automated paper review with related work retrieval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Review a paper from JSON file
  python -m src.cli --paper paper.json --output review.json

  # Review with custom settings
  python -m src.cli --paper paper.json --max-related-papers 15 --review-format detailed

  # Review with date filter
  python -m src.cli --paper paper.json --publication-date-range "2020:"
        """
    )
    
    parser.add_argument(
        "--paper",
        type=str,
        required=True,
        help="Path to paper JSON file or paper title",
    )
    parser.add_argument(
        "--vllm-url",
        type=str,
        default=None,
        help="vLLM server URL (overrides config, default: http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--llm-config",
        type=str,
        default=None,
        help="Path to LLM service config YAML file (default: configs/llm_service_config.yaml)",
    )
    parser.add_argument(
        "--asta-api-key",
        type=str,
        default=None,
        help="Asta API key (or set ASTA_API_KEY env var)",
    )
    parser.add_argument(
        "--reranker-model",
        type=str,
        default=None,
        help="Reranker model path (optional, e.g., 'OpenScholar/OpenScholar_Reranker')",
    )
    parser.add_argument(
        "--max-related-papers",
        type=int,
        default=10,
        help="Maximum number of related papers to retrieve (default: 10)",
    )
    parser.add_argument(
        "--publication-date-range",
        type=str,
        default=None,
        help="Publication date range filter (e.g., '2020:' for papers from 2020 onwards)",
    )
    parser.add_argument(
        "--venues",
        type=str,
        default=None,
        help="Venue filter (comma-separated, e.g., 'ICLR,NeurIPS')",
    )
    parser.add_argument(
        "--review-format",
        type=str,
        choices=["detailed", "summary", "structured"],
        default="detailed",
        help="Review format (default: detailed)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: print to stdout)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print intermediate outputs",
    )
    
    args = parser.parse_args()
    
    # Get API key
    asta_api_key = args.asta_api_key or os.environ.get("ASTA_API_KEY")
    if not asta_api_key:
        print("Warning: ASTA_API_KEY not set. Some features may not work.")
        print("Set it via --asta-api-key or ASTA_API_KEY environment variable.")
    
    # Initialize services
    if args.verbose:
        print("=" * 60)
        print("Initializing services...")
        print("=" * 60)
    
    reviewer = create_reviewer_pipeline(
        vllm_base_url=args.vllm_url,
        asta_api_key=asta_api_key,
        reranker_model=args.reranker_model,
        max_related_papers=args.max_related_papers,
        llm_config_file=args.llm_config,
    )
    
    if args.verbose:
        print("Services initialized successfully.\n")
    
    # Load paper data
    paper_path = Path(args.paper)
    if paper_path.exists():
        if args.verbose:
            print(f"Loading paper from: {paper_path}")
        paper_data = load_paper_from_file(str(paper_path))
        title = paper_data.get("title", "")
        abstract = paper_data.get("abstract", "")
        content = paper_data.get("content") or paper_data.get("text", "")
        keywords = paper_data.get("keywords", [])
    else:
        # Assume it's just a title
        if args.verbose:
            print(f"Using '{args.paper}' as paper title")
        title = args.paper
        abstract = ""
        content = None
        keywords = None
    
    # Generate review
    if args.verbose:
        print(f"\nReviewing paper: {title}\n")
        print("=" * 60)
        print("PIPELINE EXECUTION")
        print("=" * 60)
    
    review = reviewer.review_paper(
        title=title,
        abstract=abstract,
        content=content,
        keywords=keywords,
        publication_date_range=args.publication_date_range,
        venues=args.venues,
        review_format=args.review_format,
    )
    
    # Output results
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(review, f, indent=2, ensure_ascii=False)
        print(f"\nReview saved to {output_path}")
    else:
        print("\n" + "=" * 60)
        print("REVIEW OUTPUT")
        print("=" * 60)
        print(json.dumps(review, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

