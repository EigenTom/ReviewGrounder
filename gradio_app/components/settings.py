"""
Settings component for the Review Grounder Gradio app.

Provides:
- Prominent API key section (always visible)
- Advanced settings accordion (collapsed by default)
"""

import gradio as gr
from typing import Tuple


# Default values for OpenAI API
DEFAULT_API_ENDPOINT = "https://api.openai.com/v1"
DEFAULT_MODEL_NAME = "gpt-5.2"


def create_api_key_section() -> Tuple[gr.Textbox, gr.Textbox]:
    """
    Create the prominent API key input section.

    Returns:
        Tuple of (api_key_in, model_name_in)
    """
    gr.HTML("""
    <div class="api-key-section">
        <div class="api-key-section-title">API Configuration Required</div>
        <div class="api-key-section-desc">
            Provide your OpenAI-compatible API key to use this tool.
            Your key is never stored and is only used for this session.
        </div>
    </div>
    """)

    api_key_in = gr.Textbox(
        label="API Key",
        type="password",
        placeholder="sk-...",
        value="",
        info="Your OpenAI or compatible API key",
    )

    model_name_in = gr.Textbox(
        label="Model",
        placeholder="gpt-5.2",
        value=DEFAULT_MODEL_NAME,
        info="Model identifier (e.g. gpt-5.2, gpt-4o)",
    )

    return api_key_in, model_name_in


def create_advanced_settings() -> Tuple[gr.Textbox, gr.Checkbox, gr.Checkbox]:
    """
    Create the advanced settings accordion.

    Returns:
        Tuple of (api_base_url_in, show_log_toggle, show_raw_json_toggle)
    """
    with gr.Accordion(
        "Advanced Settings",
        open=False,
        elem_classes=["advanced-settings"],
    ):
        api_base_url_in = gr.Textbox(
            label="LLM Endpoint",
            placeholder="https://api.openai.com/v1",
            value=DEFAULT_API_ENDPOINT,
            info="Base URL for your LLM API (OpenAI, OpenRouter, local vLLM, etc.)",
        )

        gr.HTML('<div style="height: 8px;"></div>')

        show_log_toggle = gr.Checkbox(
            label="Show pipeline log",
            value=True,
            info="Display the terminal log during processing",
        )

        show_raw_json_toggle = gr.Checkbox(
            label="Show raw JSON output",
            value=False,
            info="Display raw JSON data in results",
        )

    return api_base_url_in, show_log_toggle, show_raw_json_toggle
