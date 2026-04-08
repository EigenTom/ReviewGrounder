"""
Utility wrappers and a minimal CLI for single-paper inference using:
- ASTA API key from environment variable `ASTA_API_KEY`
- OpenAI/OpenRouter endpoint and key from environment variables

This module is designed to be imported by Gradio or other web frontends,
while still remaining executable as a standalone CLI tool for debugging.
"""

import argparse
import logging
import sys
from pathlib import Path
import json
from typing import Any, Dict, Iterator, Optional

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.reviewer_agent.single_paper_inference import (
    extract_text_from_pdf,
    _split_paper_latex_sections,
    _init_single_paper_pipeline,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_single_paper_review_from_pdf(
    pdf_path: str,
    *,
    enable_logging: bool = True,
    verbose: bool = True,
    api_base_url: str | None = None,
    api_key: str | None = None,
    model_name: str | None = None,
) -> dict:
    """
    High-level utility to run the single-paper review pipeline on a PDF path.

    This is the main entry point intended to be called by Gradio or other UIs.
    It delegates to `review_single_paper_from_pdf` which uses:
    - ASTA API key from `ASTA_API_KEY`
    - LLM settings and OpenAI/OpenRouter keys from environment/config files,
      but can be overridden via `api_base_url`, `api_key`, and `model_name`.
    """
    pdf_path = str(Path(pdf_path).expanduser())
    logger.info(f"Running single-paper review for PDF: {pdf_path}")
    # Keep the original one-shot behavior for backward compatibility.
    # For true streaming updates, use `run_single_paper_review_from_pdf_stepwise`.
    from src.reviewer_agent.single_paper_inference import review_single_paper_from_pdf

    review = review_single_paper_from_pdf(
        pdf_path,
        enable_logging=enable_logging,
        verbose=verbose,
        gpt_api_key=api_key,
        gpt_base_url=api_base_url,
        gpt_model_name=model_name,
    )
    return review


def _normalize_base_url(base_url: Optional[str]) -> Optional[str]:
    """
    Normalize an OpenAI-compatible base_url.

    Your local gateway expects requests at:
        http://localhost:8000/chat/completions

    The OpenAI client will append `/chat/completions` to whatever `base_url`
    we pass in. That means:
        base_url = "http://localhost:8000"
        -> "http://localhost:8000/chat/completions"  ✅

    If a user accidentally includes `/chat/completions` in the textbox, we
    strip that suffix so the final URL is still correct.
    """
    if not base_url:
        return None
    u = base_url.strip()
    if not u:
        return None

    # Strip trailing slash for normalization
    u = u.rstrip("/")

    # If user pasted the full path (…/chat/completions), strip it back to the host.
    if u.endswith("/chat/completions"):
        u = u[: -len("/chat/completions")]

    return u


def run_single_paper_review_from_pdf_stepwise(
    pdf_path: str,
    *,
    enable_logging: bool = True,
    verbose: bool = True,
    api_base_url: str | None = None,
    api_key: str | None = None,
    model_name: str | None = None,
) -> Iterator[Dict[str, Any]]:
    """
    Stepwise (streamable) single-paper pipeline.

    Yields dict events like:
      {"stage": "extract_pdf", ...}
      {"stage": "initial_review", "initial_review": {...}}
      {"stage": "results_analysis", "results_analyzer_json": "..."}
      {"stage": "insights", "insight_miner_json": "..."}
      {"stage": "related_work", "related_work_json_list": [...], "search_keywords": [...]}
      {"stage": "final", "review": {...}}
    """
    pdf_path = str(Path(pdf_path).expanduser())
    yield {"stage": "extract_pdf", "pdf_path": pdf_path}

    paper_text = extract_text_from_pdf(pdf_path)
    yield {"stage": "parsed_pdf_text", "text_len": len(paper_text)}

    sections = _split_paper_latex_sections(paper_text)
    title = (sections.get("title") or "").strip()
    abstract = (sections.get("abstract") or "").strip()
    content = (sections.get("content") or "").strip()
    yield {"stage": "parsed_sections", "title": title, "abstract": abstract}

    reviewer, refiner, related_work_searcher, paper_results_analyzer, paper_insight_miner = (
        _init_single_paper_pipeline(
            enable_logging=enable_logging,
            use_test_llm=False,
            gpt_api_key=api_key,
            gpt_base_url=api_base_url,
            gpt_model_name=model_name,
        )
    )

    # Step 1: initial draft (reviewer)
    initial_review = reviewer.review_paper(
        title=title,
        abstract=abstract,
        content=content,
        keywords=None,
        review_format="ai_researcher",
        auto_save_log=False,
        verbose=verbose,
    )
    yield {"stage": "initial_review", "initial_review": initial_review}

    # Helper: format initial review for analyzers.
    try:
        initial_review_text = (
            refiner._format_review_dict(initial_review, "detailed")
            if hasattr(refiner, "_format_review_dict")
            else str(initial_review)
        )
    except Exception:
        initial_review_text = str(initial_review)

    # Step 2a: results analyzer
    results_analyzer_json = None
    if paper_results_analyzer and content:
        try:
            results_analyzer_json = paper_results_analyzer.analyze_paper_results(
                content, initial_review_text
            )
        except Exception as e:
            results_analyzer_json = None
            yield {"stage": "results_analysis_error", "error": str(e)}
    yield {"stage": "results_analysis", "results_analyzer_json": results_analyzer_json}

    # Step 2b: insight miner
    insight_miner_json = None
    if paper_insight_miner and content:
        try:
            insight_miner_json = paper_insight_miner.mine_paper_insights(
                content, initial_review_text
            )
        except Exception as e:
            insight_miner_json = None
            yield {"stage": "insights_error", "error": str(e)}
    yield {"stage": "insights", "insight_miner_json": insight_miner_json}

    # Step 2c: related work (structured list)
    related_work_list = []
    search_keywords = None
    if related_work_searcher:
        try:
            related_work_list = related_work_searcher.generate_related_work_json_list(
                title=title,
                abstract=abstract,
                content=content,
                keywords=None,
                publication_date_range=None,
                venues=None,
            )
            search_keywords = getattr(related_work_searcher, "last_keywords", None)
        except Exception as e:
            related_work_list = []
            yield {"stage": "related_work_error", "error": str(e)}
    yield {
        "stage": "related_work",
        "related_work_json_list": related_work_list,
        "search_keywords": search_keywords,
    }

    # Step 3: refine (final)
    related_work_json_str = json.dumps(related_work_list, ensure_ascii=False)
    refined = refiner.refine_review(
        initial_review=initial_review,
        insight_miner_json=insight_miner_json,
        results_analyzer_json=results_analyzer_json,
        related_work_json_list=related_work_json_str,
        title=title,
        abstract=abstract,
        content=content,
        review_format="detailed",
        verbose=verbose,
    )
    if search_keywords is not None:
        refined["search_keywords"] = search_keywords

    yield {"stage": "final", "review": refined}


def main() -> None:
    """
    Simple CLI wrapper mainly for local debugging.

    Example:
        python -m gradio.test_single_paper_openrouter \\
            --pdf /path/to/paper.pdf
    """
    parser = argparse.ArgumentParser(
        description="Run single-paper review on a PDF file."
    )
    parser.add_argument(
        "--pdf",
        type=str,
        required=True,
        help="Path to the PDF file to review.",
    )
    parser.add_argument(
        "--no-logging",
        action="store_true",
        help="Disable on-disk logging for this run.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce console output verbosity.",
    )
    args = parser.parse_args()

    from time import time

    start_time = time()
    review = run_single_paper_review_from_pdf(
        args.pdf,
        enable_logging=not args.no_logging,
        verbose=not args.quiet,
    )
    end_time = time()

    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print("\n=== Review keys ===")
    print(list(review.keys()))
    print("\n=== Review Markdown ===")
    print(review.get("review_markdown", ""))


if __name__ == "__main__":
    main()


