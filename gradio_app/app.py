"""
Review Grounder - Gradio App

Local development entry point.
This module orchestrates the UI components and handles the review pipeline.
"""

from __future__ import annotations

import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Tuple, Iterator

import gradio as gr

# Import utility for running the review pipeline
from utils_single_paper_inference import (
    run_single_paper_review_from_pdf_stepwise,
)

# Import UI components
from components import (
    get_custom_css,
    create_header,
    create_upload_section,
    create_api_key_section,
    create_advanced_settings,
    create_results_panel,
    generate_progress_html,
    format_initial_review_html,
    format_related_work_html,
    format_results_html,
    format_insights_html,
    format_final_review_markdown,
    format_raw_json,
)


# ============================================================================
# App Configuration
# ============================================================================

APP_TITLE = "Review Grounder"


def _raw_json_md_to_file(raw_json_md: str) -> str:
    """Extract JSON from markdown code block and write to a temp file."""
    if not raw_json_md or not raw_json_md.strip():
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            f.write("{}")
        return f.name
    text = raw_json_md.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        text = match.group(1).strip()
    fd, path = tempfile.mkstemp(suffix=".json", prefix="review_")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    return path


# ============================================================================
# Environment Check
# ============================================================================

def _check_env() -> Tuple[bool, str]:
    """Check for required environment variables."""
    missing = []
    if not os.environ.get("ASTA_API_KEY"):
        missing.append("ASTA_API_KEY")

    if missing:
        return False, (
            "Missing environment variables: "
            + ", ".join(missing)
            + ".\nPlease configure them in your Hugging Face Space settings."
        )
    return True, "Environment variables detected correctly."


# ============================================================================
# Pipeline step indices (must match PIPELINE_STEP_DEFS order)
# ============================================================================
STEP_EXTRACT = 0
STEP_DRAFT   = 1
STEP_RESULTS = 2
STEP_INSIGHTS = 3
STEP_RELATED = 4
STEP_REFINE  = 5


# ============================================================================
# Review Pipeline Handler
# ============================================================================

def review_pdf_file(
    file_obj,
    api_base_url: str,
    api_key: str,
    model_name: str,
    show_log: bool,
    show_raw_json: bool,
):
    """
    Main callback: process PDF through the review pipeline with real-time updates.
    Yields 11-element tuples for all output components including the progress bar.
    """
    log_lines: list[str] = []
    current_step = -1

    def _log(msg: str) -> None:
        log_lines.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

    def _log_text() -> str:
        return "\n".join(log_lines) if log_lines else ""

    def _final_update(md_content: str):
        if not md_content or md_content.strip() == "":
            return "*Upload a paper and run the pipeline to see the final review here.*"
        return md_content

    def _yield_state(initial, related_html, results_html, insights_html,
                     final_up, raw_json, interactive=False, finished=False):
        return (
            _log_text(), initial, related_html, results_html,
            insights_html, final_up, raw_json,
            generate_progress_html(active_step=current_step, finished=finished),
            gr.update(interactive=interactive),
            gr.update(visible=show_log),
            gr.update(visible=show_raw_json),
        )

    # Validate file upload
    if file_obj is None:
        gr.Warning("Please upload a PDF file to start the review.")
        _log("Please upload a PDF file to start the review.")
        yield _yield_state("", "", "", "", _final_update(""), "", interactive=True)
        return

    # Check environment
    ok, msg = _check_env()
    if not ok:
        gr.Error(msg)
        _log(f"ERROR: {msg}")
        yield _yield_state("", "", "", "", _final_update(""), "", interactive=True)
        return

    # Start pipeline
    _log("Pipeline started.")
    current_step = STEP_EXTRACT
    yield _yield_state("", "", "", "", _final_update(""), "")

    try:
        # Normalize file path
        if isinstance(file_obj, dict) and "name" in file_obj:
            src_path = Path(file_obj["name"])
        else:
            src_path = Path(getattr(file_obj, "name", "") or str(file_obj))

        if not src_path or not src_path.exists():
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp_path = Path(tmp.name)
                if hasattr(file_obj, "read"):
                    tmp.write(file_obj.read())
                src_path = tmp_path

        _log(f"Extracting text from PDF: {src_path.name}...")
        yield _yield_state("", "", "", "", _final_update(""), "")

        initial = ""
        related_html = ""
        results_html = ""
        insights_html = ""
        final_html = ""
        raw_json = ""

        for ev in run_single_paper_review_from_pdf_stepwise(
            str(src_path),
            api_base_url=api_base_url or None,
            api_key=api_key or None,
            model_name=model_name or None,
            enable_logging=True,
            verbose=True,
        ):
            stage = ev.get("stage")

            # Handle step-level errors
            if stage == "results_analysis_error":
                err = ev.get("error", "Unknown error")
                gr.Warning(f"Results analysis failed: {err}")
                _log(f"WARN: Results analysis failed: {err}")
                yield _yield_state(initial, related_html, results_html,
                                   insights_html, _final_update(final_html), raw_json)
                continue

            if stage == "insights_error":
                err = ev.get("error", "Unknown error")
                gr.Warning(f"Insight mining failed: {err}")
                _log(f"WARN: Insight mining failed: {err}")
                yield _yield_state(initial, related_html, results_html,
                                   insights_html, _final_update(final_html), raw_json)
                continue

            if stage == "related_work_error":
                err = ev.get("error", "Unknown error")
                gr.Warning(f"Related work search failed: {err}")
                _log(f"WARN: Related work search failed: {err}")
                yield _yield_state(initial, related_html, results_html,
                                   insights_html, _final_update(final_html), raw_json)
                continue

            if stage == "extract_pdf":
                current_step = STEP_EXTRACT
                _log(f"Extracting text from PDF: {src_path.name}...")

            elif stage == "parsed_pdf_text":
                current_step = STEP_DRAFT
                _log("Step 0/5  PDF text extraction ---- done")
                _log("Step 1/5  Initial review draft --- started")

            elif stage == "initial_review":
                current_step = STEP_RESULTS
                tmp = {"initial_review": ev.get("initial_review", {})}
                tmp["title"] = ev.get("title") or tmp["initial_review"].get("title")
                tmp["abstract"] = ev.get("abstract") or tmp["initial_review"].get("abstract")
                initial = format_initial_review_html(tmp)
                _log("Step 1/5  Initial review draft --- done")
                _log("Step 2/5  Results analysis ------ started")

            elif stage == "results_analysis":
                current_step = STEP_INSIGHTS
                tmp = {"results_analyzer_json": ev.get("results_analyzer_json")}
                results_html = format_results_html(tmp)
                _log("Step 2/5  Results analysis ------ done")
                _log("Step 3/5  Insight mining -------- started")

            elif stage == "insights":
                current_step = STEP_RELATED
                tmp = {"insight_miner_json": ev.get("insight_miner_json")}
                insights_html = format_insights_html(tmp)
                _log("Step 3/5  Insight mining -------- done")
                _log("Step 4/5  Related work ---------- started")

            elif stage == "related_work":
                current_step = STEP_REFINE
                tmp = {
                    "related_work_json_list": ev.get("related_work_json_list"),
                    "search_keywords": ev.get("search_keywords"),
                }
                related_html = format_related_work_html(tmp)
                _log("Step 4/5  Related work ---------- done")
                _log("Step 5/5  Final refinement ------ started")

            elif stage == "final":
                review = ev.get("review", {}) or {}
                initial = format_initial_review_html(review)
                related_html = format_related_work_html(review) if not related_html else related_html
                results_html = format_results_html(review) if not results_html else results_html
                insights_html = format_insights_html(review) if not insights_html else insights_html
                final_html = format_final_review_markdown(review)
                raw_json = format_raw_json(review)
                _log("Step 5/5  Final refinement ------ done")
                _log(f"Review complete: {src_path.name}")

            else:
                _log(f"Working... ({stage})")

            yield _yield_state(initial, related_html, results_html,
                               insights_html, _final_update(final_html), raw_json)

        # All done
        yield _yield_state(initial, related_html, results_html,
                           insights_html, _final_update(final_html), raw_json,
                           interactive=True, finished=True)

    except Exception as e:
        import traceback
        error_msg = f"ERROR: {str(e)}"
        error_details = traceback.format_exc()
        gr.Error(f"{error_msg}\n\nDetails: {error_details[:500]}")
        _log(error_msg)
        yield _yield_state("", "", "", "", _final_update(""), "", interactive=True)


# ============================================================================
# Build the Gradio App
# ============================================================================

_theme = gr.themes.Base(
    font=[gr.themes.GoogleFont("DM Sans"), "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
)

with gr.Blocks(title=APP_TITLE, css=get_custom_css(), theme=_theme) as demo:

    # Header
    create_header()

    # Main layout: two columns
    with gr.Row(equal_height=False):

        # Left column: API key, upload, settings
        with gr.Column(scale=2, elem_classes=["panel-card"]):
            # Prominent API key section
            api_key_in, model_name_in = create_api_key_section()

            gr.HTML('<div style="height: 8px;"></div>')

            # Upload section
            pdf_input, run_button = create_upload_section()

            # Advanced settings (collapsed)
            (
                api_base_url_in,
                show_log_toggle,
                show_raw_json_toggle,
            ) = create_advanced_settings()

        # Right column: results
        with gr.Column(scale=3, elem_classes=["panel-card", "results-panel"]):
            (
                initial_html,
                results_html,
                insights_html,
                related_html,
                final_html,
                progress_html,
                status_output,
                raw_json_md,
                log_accordion,
                raw_json_tab,
                download_json_btn,
            ) = create_results_panel(show_log=True, show_raw_json=False)

    # Toggle visibility of log accordion
    show_log_toggle.change(
        fn=lambda x: gr.update(visible=x),
        inputs=[show_log_toggle],
        outputs=[log_accordion],
    )

    # Toggle visibility of raw JSON tab
    show_raw_json_toggle.change(
        fn=lambda x: gr.update(visible=x),
        inputs=[show_raw_json_toggle],
        outputs=[raw_json_tab],
    )

    # Download raw JSON as file
    download_json_btn.click(
        fn=_raw_json_md_to_file,
        inputs=[raw_json_md],
        outputs=[download_json_btn],
    )

    # Main review button click handler
    run_button.click(
        fn=review_pdf_file,
        inputs=[
            pdf_input,
            api_base_url_in,
            api_key_in,
            model_name_in,
            show_log_toggle,
            show_raw_json_toggle,
        ],
        outputs=[
            status_output,
            initial_html,
            related_html,
            results_html,
            insights_html,
            final_html,
            raw_json_md,
            progress_html,
            run_button,
            log_accordion,
            raw_json_tab,
        ],
    )

    # Footer
    gr.HTML("""
    <div class="app-footer">
        <p>Review Grounder &middot; AI-Powered Research Paper Review</p>
    </div>
    """)


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    demo.launch()
