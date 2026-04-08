"""
Header component for the Review Grounder Gradio app.

Editorial-style header with serif title and mono eyebrow,
following the PreScouter design system.
"""

import gradio as gr


def create_header() -> None:
    """
    Create the app header with editorial typography and privacy notice.
    """
    gr.HTML("""
    <div class="app-header">
        <div class="app-header-eyebrow">SKY Lab @ TAMU Presents</div>
        <h1 class="app-header-title">Review Grounder</h1>
        <p class="app-header-subtitle">
            Upload your research paper and receive a comprehensive, evidence-grounded review
            powered by a fact-grounded multi-agent analysis pipeline.
        </p>
        <div class="privacy-notice">
            <strong>Privacy</strong> &mdash; This is an anonymous demonstration. We do not save your PDF, paper information, or any uploaded content.
        </div>
    </div>
    """)
