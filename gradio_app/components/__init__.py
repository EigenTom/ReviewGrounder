"""
Components package for the Review Grounder Gradio app.
"""

from .styles import get_custom_css
from .formatters import (
    safe_json_parse,
    format_overview,
    format_initial_review,
    format_initial_review_html,
    format_initial_review_markdown,
    format_related_work_html,
    format_results_html,
    format_insights_html,
    format_final_review,
    format_final_review_html,
    format_final_review_markdown,
    format_raw_json,
)
from .header import create_header
from .upload_section import (
    create_how_it_works,
    create_upload_area,
    create_action_buttons,
    create_upload_section,
)
from .settings import (
    create_api_key_section,
    create_advanced_settings,
    DEFAULT_API_ENDPOINT,
    DEFAULT_MODEL_NAME,
)
from .results_panel import (
    create_results_placeholder,
    create_results_panel,
    generate_progress_html,
)


__all__ = [
    # Styles
    "get_custom_css",
    # Formatters
    "safe_json_parse",
    "format_overview",
    "format_initial_review",
    "format_initial_review_html",
    "format_initial_review_markdown",
    "format_related_work_html",
    "format_results_html",
    "format_insights_html",
    "format_final_review",
    "format_final_review_html",
    "format_final_review_markdown",
    "format_raw_json",
    # Header
    "create_header",
    # Upload section
    "create_how_it_works",
    "create_upload_area",
    "create_action_buttons",
    "create_upload_section",
    # Settings
    "create_api_key_section",
    "create_advanced_settings",
    "DEFAULT_API_ENDPOINT",
    "DEFAULT_MODEL_NAME",
    # Results panel
    "create_results_placeholder",
    "create_results_panel",
    "generate_progress_html",
]
