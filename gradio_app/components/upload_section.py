"""
Upload section component for the Review Grounder Gradio app.

Provides the left panel with:
- "How it works" instructions
- PDF upload area
- Action button
"""

import gradio as gr
from typing import Tuple


def create_how_it_works() -> None:
    """Create the instruction steps section."""
    gr.HTML("""
    <div class="how-it-works">
        <div class="how-it-works-title">How It Works</div>
        <div class="step-item">
            <span class="step-number">1</span>
            <span>Enter your API key and upload a research paper (PDF)</span>
        </div>
        <div class="step-item">
            <span class="step-number">2</span>
            <span>Click <strong>Generate Review</strong> to start the pipeline</span>
        </div>
        <div class="step-item">
            <span class="step-number">3</span>
            <span>Watch the multi-agent analysis in the live terminal</span>
        </div>
        <div class="step-item">
            <span class="step-number">4</span>
            <span>Receive a comprehensive, evidence-grounded review</span>
        </div>
    </div>
    """)


def create_upload_area() -> gr.File:
    """Create the PDF upload area."""
    pdf_input = gr.File(
        label="",
        file_types=[".pdf"],
        type="filepath",
        elem_classes=["upload-area", "file-upload-minimal"],
        elem_id="pdf-upload",
        show_label=False,
    )

    gr.HTML("""
    <div class="upload-hint" style="text-align: center; margin-top: -8px; margin-bottom: 12px;">
        PDF format &middot; max 10 MB
    </div>
    """)

    return pdf_input


def create_action_buttons() -> gr.Button:
    """Create the primary generate button."""
    run_button = gr.Button(
        "Generate Review",
        variant="primary",
        elem_classes=["primary-btn"],
    )
    return run_button


def create_upload_section() -> Tuple[gr.File, gr.Button]:
    """
    Create the complete upload section.

    Returns:
        Tuple of (pdf_input, run_button)
    """
    gr.HTML('<div class="panel-eyebrow">Upload</div>')
    gr.HTML('<div class="panel-title">Your Research Paper</div>')

    create_how_it_works()
    pdf_input = create_upload_area()
    run_button = create_action_buttons()

    return pdf_input, run_button
