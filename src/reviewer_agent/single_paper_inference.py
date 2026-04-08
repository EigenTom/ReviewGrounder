"""
Single-paper inference utilities for Hugging Face Spaces or other web backends.

This module provides:
- PDF-to-text extraction using pdfminer
- Single-paper review pipeline that accepts either PDF paths or pre-extracted text
- Instantiates the OpenScholar reranker model directly (local mode)
- Uses ASTA and GPT API keys from environment variables
- Reuses the existing review pipeline with refiner.
"""

import os
import re
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from io import StringIO

try:
    from pdfminer.converter import TextConverter
    from pdfminer.layout import LAParams
    from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
    from pdfminer.pdfpage import PDFPage
    HAS_PDFMINER = True
except ImportError:
    HAS_PDFMINER = False
    TextConverter = None
    LAParams = None
    PDFResourceManager = None
    PDFPageInterpreter = None
    PDFPage = None

from .main_pipeline import (
    create_review_pipeline_with_refiner,
    review_paper_with_refiner,
)
from shared.utils.llm_service import LLMService, ChatMessage
from shared.utils.gpt_service import GPTService

try:
    from FlagEmbedding import FlagReranker
except ImportError:  # pragma: no cover - optional dependency
    FlagReranker = None

logger = logging.getLogger(__name__)

# Project root (same convention as other modules)
project_root = Path(__file__).parent.parent.parent

# Persistent, module-level cache so HF Space workers can reuse heavy objects
_PERSISTENT_RERANKER: Optional[Any] = None
_PERSISTENT_PIPELINE: Optional[
    Tuple[Any, Any, Any, Any, Any]
] = None  # reviewer, refiner, related_work_searcher, paper_results_analyzer, paper_insight_miner


class TestLLMService(LLMService):
    """
    Simple test LLM service that always returns a fixed string.
    This is used to avoid consuming real LLM quota during integration tests.
    """

    def __init__(self, response_text: str = "test_text") -> None:
        self.response_text = response_text

    def generate(
        self,
        messages: list[ChatMessage] | list[Dict[str, str]],
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        max_tokens: int = 16384,
        presence_penalty: float = 0.0,
        **kwargs: Any,
    ) -> str:
        return self.response_text

    def stream_generate(
        self,
        messages: list[ChatMessage] | list[Dict[str, str]],
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        max_tokens: int = 16384,
        presence_penalty: float = 0.0,
        **kwargs: Any,
    ):
        yield self.response_text


def _load_persistent_reranker() -> Optional[Any]:
    """
    Load the OpenScholar reranker once and reuse it.

    Uses the OpenScholar model specified in the shared config by default:
    - "OpenScholar/OpenScholar_Reranker"
    """
    global _PERSISTENT_RERANKER

    if _PERSISTENT_RERANKER is not None:
        return _PERSISTENT_RERANKER

    if FlagReranker is None:
        logger.warning(
            "FlagReranker (FlagEmbedding) not available. "
            "Install it to enable local reranking."
        )
        return None

    # Default OpenScholar reranker model
    model_name = os.environ.get(
        "OPENSCHOLAR_RERANKER_MODEL", "OpenScholar/OpenScholar_Reranker"
    )
    logger.info(f"Loading OpenScholar reranker model: {model_name}")
    _PERSISTENT_RERANKER = FlagReranker(model_name, use_fp16=True)
    logger.info("OpenScholar reranker loaded successfully and cached.")
    return _PERSISTENT_RERANKER


def _init_single_paper_pipeline(
    enable_logging: bool = False,
    use_test_llm: bool = False,
    gpt_api_key: Optional[str] = None,
    gpt_base_url: Optional[str] = None,
    gpt_model_name: Optional[str] = None,
) -> Tuple[Any, Any, Any, Any, Any]:
    """
    Initialize and cache the full review pipeline for single-paper inference.

    - Forces direct reranker mode (local model), ignoring reranker API endpoints
    - Reads ASTA API key from the environment variable ASTA_API_KEY
    - LLM configuration (vLLM/GPT) is controlled by shared/configs/llm_service_config.yaml
      and the OPENAI_API_KEY environment variable for GPT-based components.
    """
    global _PERSISTENT_PIPELINE
    
    # If we're using the default env/config-based LLM, reuse cached pipeline.
    # For test LLM or custom GPT overrides, always rebuild so changes take effect.
    if (
        _PERSISTENT_PIPELINE is not None
        and not use_test_llm
        and not any([gpt_api_key, gpt_base_url, gpt_model_name])
    ):
        # When using real LLMs, reuse the cached pipeline. For the test LLM,
        # we rebuild so that switching the flag takes effect immediately.
        return _PERSISTENT_PIPELINE

    reranker = _load_persistent_reranker()
    
    # Allow overriding to a local test LLM via environment variable or flag.
    env_use_test = os.environ.get("USE_TEST_LLM", "").lower() in {"1", "true", "yes"}
    effective_use_test_llm = use_test_llm or env_use_test
    
    test_llm: Optional[LLMService] = None
    if effective_use_test_llm:
        logger.info("Using TestLLMService (returns fixed 'test_text') for all LLM calls.")
        test_llm = TestLLMService()
    
    # Optional user-specified GPT endpoint / key / model.
    custom_llm: Optional[LLMService] = None
    if not effective_use_test_llm and any([gpt_api_key, gpt_base_url, gpt_model_name]):
        model_name = (
            gpt_model_name
            or os.environ.get("SINGLE_PAPER_GPT_MODEL")
            or "gpt-4o"
        )
        logger.info(
            "Using custom GPTService for single-paper pipeline "
            f"(model={model_name}, base_url={gpt_base_url or os.environ.get('OPENAI_BASE_URL')})"
        )
        custom_llm = GPTService(
            api_key=gpt_api_key,
            model_name=model_name,
            base_url=gpt_base_url,
        )

    pipeline_args: Dict[str, Any] = dict(
        # Use env-based keys by default; configs should not contain secrets.
        asta_api_key=os.environ.get("ASTA_API_KEY"),
        reranker=reranker,
        # Ensure we use the local reranker even if API endpoints exist in the config.
        force_direct_reranker=True,
        enable_logging=enable_logging,
        log_dir=str(project_root / "runs") if enable_logging else None,
    )

    if test_llm is not None:
        # Use the same lightweight test LLM for all components to avoid real API calls.
        pipeline_args.update(
            dict(
                keyword_llm_service=test_llm,
                summarizer_llm_service=test_llm,
                reviewer_llm_service=test_llm,
                refiner_llm_service=test_llm,
                results_summarizer_llm_service=test_llm,
                insight_miner_llm_service=test_llm,
            )
        )
    elif custom_llm is not None:
        # Use the same custom GPTService for all components so that user-provided
        # endpoint / key / model are respected everywhere.
        pipeline_args.update(
            dict(
                keyword_llm_service=custom_llm,
                summarizer_llm_service=custom_llm,
                reviewer_llm_service=custom_llm,
                refiner_llm_service=custom_llm,
                results_summarizer_llm_service=custom_llm,
                insight_miner_llm_service=custom_llm,
            )
        )
    
    reviewer, refiner, related_work_searcher, paper_results_analyzer, paper_insight_miner = (
        create_review_pipeline_with_refiner(**pipeline_args)
    )
    
    pipeline_tuple = (
        reviewer,
        refiner,
        related_work_searcher,
        paper_results_analyzer,
        paper_insight_miner,
    )
    
    # Only cache the default env-based pipeline; custom/test pipelines are per-call.
    if not (
        effective_use_test_llm or any([gpt_api_key, gpt_base_url, gpt_model_name])
    ):
        _PERSISTENT_PIPELINE = pipeline_tuple
    
    return pipeline_tuple


def _normalize_pdf_text(text: str) -> str:
    """Replace non-ASCII typographic characters with their ASCII equivalents.

    PDF renderers frequently emit smart quotes, em/en dashes, ligatures, and
    other Unicode codepoints that cause ``UnicodeEncodeError`` when downstream
    code (or a dependency) attempts ASCII serialization.
    """
    _REPLACEMENTS = {
        # Smart / curly quotes
        '\u2018': "'",   # left single quotation mark
        '\u2019': "'",   # right single quotation mark
        '\u201A': "'",   # single low-9 quotation mark
        '\u201B': "'",   # single high-reversed-9 quotation mark
        '\u201C': '"',   # left double quotation mark
        '\u201D': '"',   # right double quotation mark
        '\u201E': '"',   # double low-9 quotation mark
        '\u201F': '"',   # double high-reversed-9 quotation mark
        '\u2039': "'",   # single left-pointing angle quotation
        '\u203A': "'",   # single right-pointing angle quotation
        '\u00AB': '"',   # left-pointing double angle quotation
        '\u00BB': '"',   # right-pointing double angle quotation
        # Dashes
        '\u2013': '-',   # en dash
        '\u2014': '--',  # em dash
        '\u2015': '--',  # horizontal bar
        '\u2012': '-',   # figure dash
        # Ellipsis
        '\u2026': '...',
        # Spaces
        '\u00A0': ' ',   # non-breaking space
        '\u2002': ' ',   # en space
        '\u2003': ' ',   # em space
        '\u2009': ' ',   # thin space
        '\u200A': ' ',   # hair space
        '\u200B': '',    # zero-width space
        '\uFEFF': '',    # BOM / zero-width no-break space
        # Common ligatures
        '\uFB01': 'fi',
        '\uFB02': 'fl',
        '\uFB03': 'ffi',
        '\uFB04': 'ffl',
        # Misc
        '\u2022': '-',   # bullet
        '\u2023': '-',   # triangular bullet
        '\u25AA': '-',   # small black square (used as bullet)
        '\u00B7': '.',   # middle dot
        '\u2027': '.',   # hyphenation point
    }
    for old, new in _REPLACEMENTS.items():
        text = text.replace(old, new)
    return text


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts text from a PDF file.

    Args:
        pdf_path: Path to the PDF file (string or Path-like)

    Returns:
        A string containing the extracted text from the PDF.

    Raises:
        ImportError: If pdfminer is not installed
        FileNotFoundError: If the PDF file doesn't exist
    """
    if not HAS_PDFMINER:
        raise ImportError(
            "pdfminer is required for PDF extraction. "
            "Install it with: pip install pdfminer.six"
        )

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    with open(pdf_path, 'rb') as file_handle:
        # Initialize a PDF resource manager to store shared resources.
        resource_manager = PDFResourceManager()

        # Set up a StringIO instance to capture the extracted text.
        text_output = StringIO()

        # Create a TextConverter to convert PDF pages to text.
        converter = TextConverter(resource_manager, text_output, laparams=LAParams())

        # Initialize a PDF page interpreter.
        interpreter = PDFPageInterpreter(resource_manager, converter)

        # Process each page in the PDF.
        for page in PDFPage.get_pages(file_handle, caching=True, check_extractable=True):
            interpreter.process_page(page)

        # Retrieve the extracted text and close the StringIO instance.
        extracted_text = text_output.getvalue()
        text_output.close()

        # Finalize the converter.
        converter.close()

    # Replace form feed characters with newlines.
    extracted_text = extracted_text.replace('\x0c', '\n')

    # Normalize non-ASCII typographic characters commonly produced by PDF
    # renderers.  These smart quotes, dashes, and ligatures can cause
    # UnicodeEncodeError downstream when libraries serialize with ASCII.
    extracted_text = _normalize_pdf_text(extracted_text)

    return extracted_text


def _split_paper_latex_sections(paper_text: str) -> Dict[str, str]:
    """
    Best-effort extraction of title, abstract, and content from LaTeX-like text.
    Mirrors the logic used in the example script but is reusable.
    """

    text = paper_text or ""

    # 1) Try LaTeX-style title/abstract first (best signal when available).
    title_match = re.search(r"\\title\{(.*?)\}", text, re.DOTALL)
    title = title_match.group(1).strip() if title_match else ""

    abstract_match = re.search(
        r"\\begin\{abstract\}(.*?)\\end\{abstract\}", text, re.DOTALL | re.IGNORECASE
    )
    abstract = abstract_match.group(1).strip() if abstract_match else ""

    # Content: everything after abstract, or full text as fallback
    content = ""
    if abstract_match:
        content_match = re.search(r"\\end\{abstract\}(.*)", text, re.DOTALL | re.IGNORECASE)
        content = content_match.group(1).strip() if content_match else ""


    if not content:
        content = text.strip()

    return {"title": title, "abstract": abstract, "content": content}


def review_single_paper_from_text(
    paper_text: str,
    *,
    title: Optional[str] = None,
    abstract: Optional[str] = None,
    keywords: Optional[Any] = None,
    review_format: Optional[str] = None,
    enable_logging: bool = False,
    verbose: Optional[bool] = None,
    use_test_llm: bool = False,
    gpt_api_key: Optional[str] = None,
    gpt_base_url: Optional[str] = None,
    gpt_model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Entry point for single-paper inference after PDF-to-text.

    Args:
        paper_text: Full text of the paper (e.g., from pdf-to-text or LaTeX source)
        title: Optional explicit title (overrides parsed title if provided)
        abstract: Optional explicit abstract (overrides parsed abstract if provided)
        keywords: Optional list of keywords
        review_format: Optional review format; if None, uses config default
        enable_logging: Whether to enable on-disk logging via ReviewLogger
        verbose: Optional verbose flag to control console output

    Returns:
        Refined review dictionary (JSON-serializable)
    """
    # Initialize or reuse the pipeline components
    reviewer, refiner, related_work_searcher, paper_results_analyzer, paper_insight_miner = (
        _init_single_paper_pipeline(
            enable_logging=enable_logging,
            use_test_llm=use_test_llm,
            gpt_api_key=gpt_api_key,
            gpt_base_url=gpt_base_url,
            gpt_model_name=gpt_model_name,
        )
    )

    # Parse sections from the raw text if not explicitly provided
    sections = _split_paper_latex_sections(paper_text)

    final_title = (title or sections["title"]).strip()
    final_abstract = (abstract or sections["abstract"]).strip()

    paper_data: Dict[str, Any] = {
        "title": final_title,
        "abstract": final_abstract,
        "content": sections["content"],
        "keywords": keywords,
    }

    if review_format is not None:
        paper_data["review_format"] = review_format

    # Run the full reviewer → refiner pipeline
    refined_review = review_paper_with_refiner(
        paper_data=paper_data,
        reviewer=reviewer,
        refiner=refiner,
        related_work_searcher=related_work_searcher,
        paper_results_analyzer=paper_results_analyzer,
        paper_insight_miner=paper_insight_miner,
        verbose=verbose,
    )

    return refined_review


def review_single_paper_from_pdf(
    pdf_path: str,
    *,
    title: Optional[str] = None,
    abstract: Optional[str] = None,
    keywords: Optional[Any] = None,
    review_format: Optional[str] = None,
    enable_logging: bool = False,
    verbose: Optional[bool] = None,
    use_test_llm: bool = False,
    gpt_api_key: Optional[str] = None,
    gpt_base_url: Optional[str] = None,
    gpt_model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Entry point for single-paper inference from a PDF file.

    This function:
    1. Extracts text from the PDF using pdfminer
    2. Calls review_single_paper_from_text with the extracted text

    Args:
        pdf_path: Path to the PDF file
        title: Optional explicit title (overrides parsed title if provided)
        abstract: Optional explicit abstract (overrides parsed abstract if provided)
        keywords: Optional list of keywords
        review_format: Optional review format; if None, uses config default
        enable_logging: Whether to enable on-disk logging via ReviewLogger
        verbose: Optional verbose flag to control console output
        use_test_llm: Whether to use TestLLMService (returns "test_text") instead of real LLM

    Returns:
        Refined review dictionary (JSON-serializable)

    Raises:
        ImportError: If pdfminer is not installed
        FileNotFoundError: If the PDF file doesn't exist
    """
    logger.info(f"Extracting text from PDF: {pdf_path}")
    paper_text = extract_text_from_pdf(pdf_path)
    logger.info(f"Extracted {len(paper_text)} characters from PDF")

    return review_single_paper_from_text(
        paper_text,
        title=title,
        abstract=abstract,
        keywords=keywords,
        review_format=review_format,
        enable_logging=enable_logging,
        verbose=verbose,
        use_test_llm=use_test_llm,
        gpt_api_key=gpt_api_key,
        gpt_base_url=gpt_base_url,
        gpt_model_name=gpt_model_name,
    )


__all__ = [
    "extract_text_from_pdf",
    "review_single_paper_from_text",
    "review_single_paper_from_pdf",
]

