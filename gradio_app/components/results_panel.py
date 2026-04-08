"""
Results panel component for the Review Grounder Gradio app.

Provides the right panel with:
- Pipeline progress indicator (dynamic, updates per step)
- Terminal log (always visible by default)
- Tabbed results for each pipeline stage
"""

import gradio as gr
from typing import Tuple


# Step definitions: (id, label)
PIPELINE_STEP_DEFS = [
    ("extract", "PDF Extract"),
    ("draft", "Initial Draft"),
    ("results", "Results"),
    ("insights", "Insights"),
    ("related", "Related Work"),
    ("refine", "Refinement"),
]


def generate_progress_html(active_step: int = -1, finished: bool = False) -> str:
    """
    Generate pipeline progress HTML with the given step highlighted.

    Args:
        active_step: 0-based index of the currently active step.
                     -1 means no step started yet.
        finished: If True, all steps are shown as done.

    Returns:
        HTML string for the progress bar.
    """
    parts = []
    parts.append('<div class="pipeline-progress">')
    parts.append('  <div class="pipeline-progress-header">Pipeline Progress</div>')
    parts.append('  <div class="pipeline-steps">')

    for i, (sid, label) in enumerate(PIPELINE_STEP_DEFS):
        if finished or i < active_step:
            cls = "pipeline-step done"
            icon = "&#10003;"  # checkmark
        elif i == active_step:
            cls = "pipeline-step active"
            icon = "&#9654;"  # right-pointing triangle (playing)
        else:
            cls = "pipeline-step"
            icon = "&#9675;"  # circle

        parts.append(
            f'    <div class="{cls}">'
            f'<span class="pipeline-step-icon">{icon}</span>'
            f'<span>{label}</span></div>'
        )

        # connector between steps
        if i < len(PIPELINE_STEP_DEFS) - 1:
            if finished or i < active_step:
                parts.append('    <div class="pipeline-connector done"></div>')
            else:
                parts.append('    <div class="pipeline-connector"></div>')

    parts.append("  </div>")
    parts.append("</div>")
    return "\n".join(parts)


def create_results_placeholder() -> str:
    return """
    <div class="results-placeholder">
        <div class="results-placeholder-icon">&#128203;</div>
        <h3>Ready to Review</h3>
        <p>Upload a research paper and click Generate Review to receive comprehensive, evidence-grounded feedback.</p>
    </div>
    """


def create_initial_draft_placeholder() -> str:
    return """
    <div class="results-placeholder">
        <div class="results-placeholder-icon">&#128221;</div>
        <h3>Initial Draft</h3>
        <p>The initial review draft will appear here once the pipeline begins.</p>
    </div>
    """


def create_results_analyzer_placeholder() -> str:
    return """
    <div class="results-placeholder">
        <div class="results-placeholder-icon">&#128200;</div>
        <h3>Results Analyzer</h3>
        <p>Experimental results analysis will appear here.</p>
    </div>
    """


def create_insights_miner_placeholder() -> str:
    return """
    <div class="results-placeholder">
        <div class="results-placeholder-icon">&#128161;</div>
        <h3>Insight Miner</h3>
        <p>Key insights extracted from the paper will appear here.</p>
    </div>
    """


def create_related_work_placeholder() -> str:
    return """
    <div class="results-placeholder">
        <div class="results-placeholder-icon">&#128218;</div>
        <h3>Related Work</h3>
        <p>Curated related papers and summaries will appear here.</p>
    </div>
    """


def create_final_review_placeholder() -> str:
    return """
    <div class="results-placeholder">
        <div class="results-placeholder-icon">&#127919;</div>
        <h3>Final Review</h3>
        <p>The refined final review — synthesizing all agent outputs — will appear here.</p>
    </div>
    """


def create_results_panel(
    show_log: bool = True,
    show_raw_json: bool = False,
) -> Tuple[gr.Markdown, gr.HTML, gr.HTML, gr.HTML, gr.Markdown, gr.HTML, gr.Textbox, gr.Markdown, gr.Accordion, gr.Tab, gr.DownloadButton]:
    """
    Create the results panel with progress indicator, terminal, and tabbed results.

    Returns:
        Tuple of (initial_html, results_html, insights_html, related_html,
                  final_html, progress_html, status_output, raw_json_md,
                  log_accordion, raw_json_tab, download_json_btn)
    """
    gr.HTML('<div class="panel-eyebrow">Results</div>')
    gr.HTML('<div class="panel-title">AI Review Output</div>')

    # Dynamic pipeline progress indicator
    progress_html = gr.HTML(
        value=generate_progress_html(active_step=-1),
        show_label=False,
    )

    # Terminal log (visible by default)
    with gr.Accordion("Terminal", open=True, visible=show_log) as log_accordion:
        gr.HTML("""
        <div class="terminal-log-header">
            <div class="terminal-log-dots">
                <span class="dot-red"></span>
                <span class="dot-yellow"></span>
                <span class="dot-green"></span>
            </div>
            <div class="terminal-log-title">Pipeline Log</div>
        </div>
        """)
        status_output = gr.Textbox(
            value="Ready. Upload a PDF and click Generate Review to start.",
            lines=6,
            max_lines=15,
            interactive=False,
            autoscroll=True,
            elem_classes=["status-log"],
            show_label=False,
        )

    # Tabbed results
    with gr.Tabs():
        with gr.Tab("Final Review", id="final"):
            gr.HTML("""
            <div class="final-review-toolbar">
                <button type="button" class="copy-final-btn" onclick="(function(){
                    var el = document.getElementById('final-review-md');
                    if (!el) { alert('Nothing to copy'); return; }
                    var md = el.querySelector('.md-content');
                    var text = (md || el).innerText || (md || el).textContent || '';
                    if (text && navigator.clipboard && navigator.clipboard.writeText) {
                        navigator.clipboard.writeText(text).then(function(){ alert('Copied to clipboard'); }).catch(function(){ alert('Copy failed'); });
                    } else { alert('Nothing to copy'); }
                })();" title="Copy review to clipboard">Copy to clipboard</button>
            </div>
            """)
            final_md = gr.Markdown(
                value="*Upload a paper and run the pipeline to see the final review here.*",
                label="Final Review",
                elem_id="final-review-md",
                latex_delimiters=[
                    {"left": "$$", "right": "$$", "display": True},
                    {"left": "$", "right": "$", "display": False},
                    {"left": "\\(", "right": "\\)", "display": False},
                    {"left": "\\[", "right": "\\]", "display": True},
                ],
            )

        with gr.Tab("Initial Draft", id="initial"):
            initial_md = gr.Markdown(
                value="*The initial review draft will appear here once the pipeline begins.*",
                label="Initial Draft",
                latex_delimiters=[
                    {"left": "$$", "right": "$$", "display": True},
                    {"left": "$", "right": "$", "display": False},
                    {"left": "\\(", "right": "\\)", "display": False},
                    {"left": "\\[", "right": "\\]", "display": True},
                ],
            )

        with gr.Tab("Results", id="results"):
            results_html = gr.HTML(
                value=create_results_analyzer_placeholder(),
                label="Results Analyzer",
            )

        with gr.Tab("Insights", id="insights"):
            insights_html = gr.HTML(
                value=create_insights_miner_placeholder(),
                label="Insight Miner",
            )

        with gr.Tab("Related Work", id="related"):
            related_html = gr.HTML(
                value=create_related_work_placeholder(),
                label="Related Work",
            )

        with gr.Tab("Raw JSON", id="raw_json", visible=show_raw_json) as raw_json_tab:
            raw_json_md = gr.Markdown(
                value="Raw JSON output will appear here.",
                label="Raw JSON",
            )
            download_json_btn = gr.DownloadButton("Download JSON")

    return (
        initial_md,
        results_html,
        insights_html,
        related_html,
        final_md,
        progress_html,
        status_output,
        raw_json_md,
        log_accordion,
        raw_json_tab,
        download_json_btn,
    )
