"""
Formatting utilities for the Review Grounder Gradio app.

This module contains all functions for formatting review data
into displayable HTML or Markdown for the UI components.
"""

from __future__ import annotations

import json
from typing import Any


def safe_json_parse(value: Any) -> Any:
    """
    Safely parse a JSON string or return the value if already parsed.
    
    Args:
        value: A JSON string or already-parsed object
        
    Returns:
        Parsed JSON object or None if parsing fails
    """
    if value is None:
        return None
    try:
        if isinstance(value, str):
            return json.loads(value)
        return value
    except Exception as e:
        # print out the exact error
        print(f"Error parsing JSON: {e}")
        
        return None


def format_overview(review: dict) -> str:
    """
    Format high-level overview: scores and keywords only.
    
    Args:
        review: The review dictionary containing scores and metadata
        
    Returns:
        Formatted Markdown string with scores and search keywords
    """
    if not review:
        return "No review data."

    scores = review.get("scores", {}) or {}
    rating = scores.get("rating") or review.get("rating")
    confidence = scores.get("confidence") or review.get("confidence")
    decision = scores.get("decision") or review.get("decision")

    parts = [
        "### Scores",
        f"- **Rating**: {rating if rating is not None else 'N/A'}",
        f"- **Confidence**: {confidence if confidence is not None else 'N/A'}",
        f"- **Decision**: {decision or 'N/A'}",
    ]

    keywords = review.get("search_keywords")
    if keywords:
        parts.append("")
        parts.append("### Search Keywords")
        parts.append("".join(f"- {k}\n" for k in keywords).rstrip())

    return "\n".join(parts)


def _escape_html(text: str) -> str:
    """Escape HTML special characters for safe display."""
    if not text:
        return ""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def format_initial_review(review: dict) -> str:
    """
    Format the initial draft review as plain text (legacy).
    Prefer format_initial_review_html for UI display.
    """
    initial = review.get("initial_review")
    if not initial:
        return "Initial draft review not available (pipeline may have failed early)."
    text = initial.get("review") or ""
    if not text:
        return json.dumps(initial, indent=2, ensure_ascii=False)
    return text


def format_initial_review_html(review: dict) -> str:
    """Legacy wrapper — delegates to format_initial_review_markdown."""
    return format_initial_review_markdown(review)


def format_initial_review_markdown(review: dict) -> str:
    """
    Format the initial draft review as Markdown for display in gr.Markdown.

    Extracts structured fields (summary, scores, strengths, weaknesses,
    questions) from the initial_review dict and renders clean markdown so
    that formatting and LaTeX pass through to Gradio's renderer.
    """
    initial = review.get("initial_review")
    if not initial:
        return "*Initial draft review not available (pipeline may have failed early).*"

    if isinstance(initial, str):
        initial = safe_json_parse(initial) or {}
        if not initial:
            return "*Initial draft data could not be parsed.*"

    # If it doesn't look like structured JSON, just return the text
    if not _looks_like_raw_json(initial):
        text = initial.get("review") or ""
        if text:
            return text
        return "*Initial draft data could not be parsed.*"

    parts: list[str] = []

    # Scores bar
    score_fields = [
        ("soundness", "Soundness"),
        ("presentation", "Presentation"),
        ("contribution", "Contribution"),
        ("rating", "Rating"),
        ("confidence", "Confidence"),
        ("decision", "Decision"),
    ]
    score_items = []
    for key, label in score_fields:
        val = initial.get(key)
        if val is not None and val != "":
            score_items.append(f"**{label}:** {val}")
    if score_items:
        parts.append(" · ".join(score_items))
        parts.append("")

    # Sections
    section_map = [
        ("summary", "Summary"),
        ("strengths", "Strengths"),
        ("weaknesses", "Weaknesses"),
        ("questions", "Questions"),
    ]
    for key, heading in section_map:
        val = initial.get(key)
        if not val:
            continue
        if isinstance(val, list):
            body = "\n".join(f"- {item}" for item in val if item)
        else:
            body = str(val)
        if body.strip():
            parts.append(f"## {heading}\n\n{body}")

    if parts:
        return "\n\n".join(parts)
    return "*Initial draft review not available.*"


def _looks_like_raw_json(obj: Any) -> bool:
    """Heuristic: dict has typical review keys (summary, strengths) then treat as structured."""
    if not isinstance(obj, dict):
        return False
    return any(k in obj for k in ("summary", "strengths", "weaknesses", "soundness", "rating"))


def _nl2br(text: str) -> str:
    """Escape HTML and convert newlines to <br> for safe display."""
    if not text:
        return ""
    return _escape_html(text).replace("\n", "<br>\n")


def format_related_work_html(review: dict) -> str:
    """
    Format related work as HTML with styled cards.
    
    Args:
        review: The review dictionary containing related_work_json_list
        
    Returns:
        HTML string with related work cards
    """
    rw = review.get("related_work_json_list")
    if not rw:
        return "<p>No related work information available.</p>"
    
    try:
        data = json.loads(rw) if isinstance(rw, str) else rw
    except Exception:
        return f"<p>Error parsing related work data: {str(rw)[:200]}</p>"
    
    if not data:
        return "<p>No related work summaries found.</p>"

    html = '<div class="related-work-container"><h3>Related Work Summaries</h3>'

    for idx, item in enumerate(data, start=1):
        summary = item.get("summary", "").strip()
        main_methods = item.get("main_methods", "").strip()
        key_findings = item.get("key_findings", "").strip()
        relation = item.get("relation", "").strip()

        html += f'<div class="related-paper-card">'
        html += f'<div class="paper-header">{idx}. {summary[:100] or "Related paper"}...</div>'
        
        if summary:
            html += f'''
            <div class="paper-field">
                <div class="paper-field-label">Summary</div>
                <div class="paper-field-value">{summary}</div>
            </div>
            '''
        if main_methods:
            html += f'''
            <div class="paper-field">
                <div class="paper-field-label">Main Methods</div>
                <div class="paper-field-value">{main_methods}</div>
            </div>
            '''
        if key_findings:
            html += f'''
            <div class="paper-field">
                <div class="paper-field-label">Key Findings</div>
                <div class="paper-field-value">{key_findings}</div>
            </div>
            '''
        if relation:
            html += f'''
            <div class="paper-field">
                <div class="paper-field-label">Relation</div>
                <div class="paper-field-value">{relation}</div>
            </div>
            '''
        html += "</div>"

    html += "</div>"
    return html


def _render_field(label: str, value: str) -> str:
    """Render a single label/value pair in the unified card style."""
    return f'''<div class="paper-field">
        <div class="paper-field-label">{_escape_html(label)}</div>
        <div class="paper-field-value">{_escape_html(value)}</div>
    </div>'''


def _render_field_list(label: str, items: list) -> str:
    """Render a list of items as a card field."""
    if not items:
        return ""
    text = "\n".join(f"- {_escape_html(str(x))}" for x in items if x)
    return f'''<div class="paper-field">
        <div class="paper-field-label">{_escape_html(label)}</div>
        <div class="paper-field-value" style="white-space:pre-wrap">{text}</div>
    </div>'''


def _render_details_list(label: str, items: list, head_keys: list, body_keys: list) -> str:
    """Render a list of dicts as collapsible details inside a card field."""
    if not items:
        return ""
    blocks = []
    for it in items:
        if isinstance(it, dict):
            head = ""
            for k in head_keys:
                if it.get(k):
                    head = it[k]
                    break
            head = head or "Item"
            body_parts = []
            for k in body_keys:
                if it.get(k):
                    body_parts.append(
                        f'<div class="paper-field-label">{k.replace("_", " ").title()}</div>'
                        f'<div class="paper-field-value">{_escape_html(it[k])}</div>'
                    )
            body = "".join(body_parts) or (
                f'<div class="paper-field-value mono">'
                f'{_escape_html(json.dumps(it, indent=2, ensure_ascii=False))}</div>'
            )
            blocks.append(
                f'<details class="card-details"><summary>{_escape_html(head)}</summary>'
                f'<div class="details-body">{body}</div></details>'
            )
        else:
            blocks.append(f'<div class="paper-field-value">{_escape_html(str(it))}</div>')
    return f'''<div class="paper-field">
        <div class="paper-field-label">{_escape_html(label)}</div>
        {"".join(blocks)}
    </div>'''


def _render_review_issues_html(issues: dict) -> str:
    """Render review issues as collapsible details in the unified card style."""
    if not issues:
        return ""
    html = ""
    issue_sections = [
        ("incorrect_or_hallucinated", "Incorrect / Hallucinated",
         ["review_claim", "what_missing", "review_text"],
         ["why_wrong", "evidence", "how_to_fix"]),
        ("missing_key_points", "Missing Key Points",
         ["what_missing", "review_claim"],
         ["why_important", "evidence"]),
        ("needs_specificity", "Needs Specificity",
         ["review_text", "review_claim"],
         ["how_to_fix", "evidence"]),
    ]
    for key, label, hkeys, bkeys in issue_sections:
        items = issues.get(key, [])
        if items:
            html += _render_details_list(label, items, hkeys, bkeys)
    return html


def _render_rewrite_suggestions_html(suggestions: list) -> str:
    """Render rewrite suggestions in the unified card style."""
    if not suggestions:
        return ""
    blocks = []
    for s in suggestions:
        if isinstance(s, dict):
            head = f"{s.get('apply_to', 'Rewrite')} · {s.get('target', '')}".strip(" ·")
            body = ""
            if s.get("suggested_text"):
                body += (f'<div class="paper-field-label">Suggested Text</div>'
                         f'<div class="paper-field-value">{_escape_html(s["suggested_text"])}</div>')
            if s.get("evidence"):
                body += (f'<div class="paper-field-label">Evidence</div>'
                         f'<div class="paper-field-value">{_escape_html(s["evidence"])}</div>')
            blocks.append(
                f'<details class="card-details"><summary>{_escape_html(head or "Rewrite suggestion")}'
                f'</summary><div class="details-body">{body}</div></details>'
            )
        else:
            blocks.append(f'<div class="paper-field-value">{_escape_html(str(s))}</div>')
    return f'''<div class="paper-field">
        <div class="paper-field-label">Rewrite Suggestions</div>
        {"".join(blocks)}
    </div>'''


def format_results_html(review: dict) -> str:
    """Format results analyzer output using the unified card style."""
    parsed = safe_json_parse(review.get("results_analyzer_json"))
    if not parsed:
        return "<p>Unable to parse the results analysis data.</p>"

    facts = parsed.get("facts", {}) if isinstance(parsed, dict) else {}
    review_issues = parsed.get("review_issues", {}) if isinstance(parsed, dict) else {}
    rewrite_suggestions = parsed.get("rewrite_suggestions", []) if isinstance(parsed, dict) else []

    html = '<div class="related-work-container"><h3>Results Analysis</h3>'

    # Facts card
    html += '<div class="related-paper-card">'
    html += '<div class="paper-header">Extracted Facts</div>'
    html += _render_field_list("Datasets", facts.get("datasets", []))
    html += _render_field_list("Metrics", facts.get("metrics", []))
    html += _render_field_list("Baselines", facts.get("baselines", []))
    html += _render_details_list(
        "Key Results",
        facts.get("key_results", []),
        head_keys=["claim"],
        body_keys=["evidence"],
    )
    html += "</div>"

    # Review issues card
    if review_issues:
        html += '<div class="related-paper-card">'
        html += '<div class="paper-header">Review Issues</div>'
        html += _render_review_issues_html(review_issues)
        html += "</div>"

    # Rewrite suggestions card
    if rewrite_suggestions:
        html += '<div class="related-paper-card">'
        html += '<div class="paper-header">Rewrite Suggestions</div>'
        html += _render_rewrite_suggestions_html(rewrite_suggestions)
        html += "</div>"

    html += "</div>"
    return html


def format_insights_html(review: dict) -> str:
    """Format insight miner output using the unified card style."""
    parsed = safe_json_parse(review.get("insight_miner_json"))
    if not parsed:
        return "<p>Unable to parse the insights data.</p>"

    facts = parsed.get("facts", {}) if isinstance(parsed, dict) else {}
    review_issues = parsed.get("review_issues", {}) if isinstance(parsed, dict) else {}
    rewrite_suggestions = parsed.get("rewrite_suggestions", []) if isinstance(parsed, dict) else []

    html = '<div class="related-work-container"><h3>Paper Insights</h3>'

    # Facts card
    html += '<div class="related-paper-card">'
    html += '<div class="paper-header">Extracted Insights</div>'

    insight_sections = [
        ("core_contributions", "Core Contributions", ["claim"], ["evidence"]),
        ("method_summary", "Method Summary", ["point"], ["evidence"]),
        ("assumptions_and_scope", "Assumptions & Scope", ["item"], ["evidence"]),
        ("novelty_claims_in_paper", "Novelty Claims", ["claim"], ["evidence"]),
    ]
    for key, label, hkeys, bkeys in insight_sections:
        items = facts.get(key, [])
        if items:
            html += _render_details_list(label, items, hkeys, bkeys)

    html += "</div>"

    # Review issues card
    if review_issues:
        html += '<div class="related-paper-card">'
        html += '<div class="paper-header">Review Issues</div>'
        html += _render_review_issues_html(review_issues)
        html += "</div>"

    # Rewrite suggestions card
    if rewrite_suggestions:
        html += '<div class="related-paper-card">'
        html += '<div class="paper-header">Rewrite Suggestions</div>'
        html += _render_rewrite_suggestions_html(rewrite_suggestions)
        html += "</div>"

    html += "</div>"
    return html


def format_final_review(review: dict) -> str:
    """
    Extract and return the final review markdown (legacy).
    Prefer format_final_review_html for UI display.
    
    Args:
        review: The review dictionary containing the final review
        
    Returns:
        The final review markdown string
    """
    return review.get("review_markdown") or review.get("review") or "Final review markdown missing."


def format_final_review_html(review: dict) -> str:
    """
    Format the final refined review as styled HTML cards (similar to Initial Draft).
    
    Tries to extract structured fields (summary, scores, strengths, weaknesses, etc.)
    from review_json or directly from review dict, then renders as HTML cards.
    Falls back to converting markdown to HTML if no structured data available.
    
    Args:
        review: The review dictionary containing the final review
        
    Returns:
        HTML string for display in gr.HTML
    """
    # Always try to keep the original markdown/text around for copy-to-clipboard
    markdown_text = review.get("review_markdown") or review.get("review") or ""
    data_attr = ""
    if markdown_text:
        # Store markdown in a data attribute on a wrapper div.
        # Use _escape_html so it's safe inside the attribute.
        safe_md = _escape_html(markdown_text)
        data_attr = f' data-markdown="{safe_md}"'

    # Try to get structured JSON first
    review_json = review.get("review_json")
    if review_json:
        # Parse if it's a string
        if isinstance(review_json, str):
            review_json = safe_json_parse(review_json)
        if review_json and isinstance(review_json, dict):
            # Use structured JSON for rendering
            inner_html = _render_final_review_from_dict(review_json, review)
            if data_attr:
                return f'<div class="final-review-wrapper"{data_attr}>{inner_html}</div>'
            return inner_html
    
    # If no review_json, try to extract structured fields directly from review dict
    # (final review might have summary, strengths, weaknesses, etc. at top level)
    if _looks_like_raw_json(review):
        inner_html = _render_final_review_from_dict(review, review)
        if data_attr:
            return f'<div class="final-review-wrapper"{data_attr}>{inner_html}</div>'
        return inner_html
    
    # Fallback: parse markdown and convert to HTML
    if markdown_text:
        inner_html = _markdown_to_html_cards(markdown_text, review)
        if data_attr:
            return f'<div class="final-review-wrapper"{data_attr}>{inner_html}</div>'
        return inner_html
    
    # No content available
    return "<p class='review-message'>Final review content not available.</p>"


def _render_final_review_from_dict(review_data: dict, full_review: dict) -> str:
    """
    Render structured review dict as HTML cards (similar to Initial Draft).
    
    Args:
        review_data: The structured review dictionary (from review_json or top-level fields)
        full_review: The complete review dict (for accessing scores, etc.)
        
    Returns:
        HTML string with formatted cards
    """
    html = '<div class="final-review-card card-grid"><div class="card"><h4>🎯 Final Review</h4>'
    
    # Summary
    summary = review_data.get("summary") or ""
    if summary:
        html += f'<div class="kv"><div class="k">Summary</div><div class="v">{_escape_html(summary)}</div></div>'
    
    # Scores (from full_review.scores or review_data)
    scores = full_review.get("scores", {}) or {}
    score_fields = [
        ("soundness", "Soundness"),
        ("presentation", "Presentation"),
        ("contribution", "Contribution"),
        ("rating", "Rating"),
        ("confidence", "Confidence"),
        ("decision", "Decision"),
    ]
    for key, label in score_fields:
        val = scores.get(key) or review_data.get(key)
        if val is not None and val != "":
            html += f'<div class="kv"><div class="k">{label}</div><div class="v">{_escape_html(str(val))}</div></div>'
    
    # Strengths
    strengths = review_data.get("strengths") or ""
    if strengths:
        if isinstance(strengths, list):
            strengths_text = "\n".join(f"• {s}" for s in strengths if s)
        else:
            strengths_text = str(strengths)
        if strengths_text:
            html += f'<div class="kv"><div class="k">Strengths</div><div class="v">{_nl2br(strengths_text)}</div></div>'
    
    # Weaknesses
    weaknesses = review_data.get("weaknesses") or ""
    if weaknesses:
        if isinstance(weaknesses, list):
            weaknesses_text = "\n".join(f"• {w}" for w in weaknesses if w)
        else:
            weaknesses_text = str(weaknesses)
        if weaknesses_text:
            html += f'<div class="kv"><div class="k">Weaknesses</div><div class="v">{_nl2br(weaknesses_text)}</div></div>'
    
    # Questions
    questions = review_data.get("questions")
    if questions:
        if isinstance(questions, list):
            q_text = "\n".join(f"• {q}" for q in questions if q)
        else:
            q_text = str(questions)
        if q_text:
            html += f'<div class="kv"><div class="k">Questions</div><div class="v">{_nl2br(q_text)}</div></div>'
    
    # # Additional sections that might be in markdown format
    # # Try to extract from markdown if structured fields are missing
    # markdown_text = full_review.get("review_markdown") or full_review.get("review") or ""
    # if markdown_text and not (summary or strengths or weaknesses):
    #     # If no structured fields, render the full markdown as HTML
    #     html += f'<div class="kv"><div class="k">Review</div><div class="v review-text">{_markdown_to_html(markdown_text)}</div></div>'
    # elif markdown_text:
    #     # If we have some structured fields, append the full markdown as additional content
    #     html += f'<div class="kv"><div class="k">Full Review</div><div class="v review-text">{_markdown_to_html(markdown_text)}</div></div>'
    
    html += "</div></div>"
    return html


def _markdown_to_html_cards(markdown_text: str, review: dict) -> str:
    """
    Convert markdown review text to HTML cards format.
    
    Args:
        markdown_text: The markdown review text
        review: The full review dict (for accessing scores)
        
    Returns:
        HTML string with formatted cards
    """
    html = '<div class="final-review-card card-grid"><div class="card"><h4>🎯 Final Review</h4>'
    
    # Add scores if available
    scores = review.get("scores", {}) or {}
    if scores:
        score_fields = [
            ("soundness", "Soundness"),
            ("presentation", "Presentation"),
            ("contribution", "Contribution"),
            ("rating", "Rating"),
            ("confidence", "Confidence"),
            ("decision", "Decision"),
        ]
        for key, label in score_fields:
            val = scores.get(key)
            if val is not None and val != "":
                html += f'<div class="kv"><div class="k">{label}</div><div class="v">{_escape_html(str(val))}</div></div>'
    
    # Convert markdown to HTML and display
    html += f'<div class="kv"><div class="k">Review</div><div class="v review-text">{_markdown_to_html(markdown_text)}</div></div>'
    
    html += "</div></div>"
    return html


def _markdown_to_html(markdown_text: str) -> str:
    """
    Convert markdown text to HTML, preserving structure.
    
    Simple conversion: handles headers, lists, bold, italic, code blocks.
    Preserves line breaks and basic formatting.
    
    Args:
        markdown_text: Markdown string
        
    Returns:
        HTML string
    """
    if not markdown_text:
        return ""
    
    lines = markdown_text.split("\n")
    html_parts = []
    in_code_block = False
    code_block_content = []
    in_list = False
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Handle code blocks
        if stripped.startswith("```"):
            if in_code_block:
                # End code block
                html_parts.append(f'<pre class="mono"><code>{_escape_html("<br>".join(code_block_content))}</code></pre>')
                code_block_content = []
                in_code_block = False
            else:
                # Start code block
                if in_list:
                    html_parts.append("</ul>")
                    in_list = False
                in_code_block = True
            continue
        
        if in_code_block:
            code_block_content.append(line)
            continue
        
        # Headers
        if stripped.startswith("### "):
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            html_parts.append(f'<h3>{_process_inline_markdown(stripped[4:])}</h3>')
        elif stripped.startswith("## "):
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            html_parts.append(f'<h2>{_process_inline_markdown(stripped[3:])}</h2>')
        elif stripped.startswith("# "):
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            html_parts.append(f'<h1>{_process_inline_markdown(stripped[2:])}</h1>')
        # Lists
        elif stripped.startswith("- ") or stripped.startswith("* ") or (stripped and stripped[0].isdigit() and ". " in stripped[:5]):
            if not in_list:
                html_parts.append("<ul>")
                in_list = True
            list_content = stripped.split(". ", 1)[-1] if ". " in stripped[:5] else stripped[2:]
            html_parts.append(f'<li>{_process_inline_markdown(list_content)}</li>')
        # Empty line
        elif not stripped:
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            html_parts.append("<br>")
        # Regular paragraph
        else:
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            html_parts.append(f'<p>{_process_inline_markdown(line)}</p>')
    
    # Close any open list
    if in_list:
        html_parts.append("</ul>")
    
    return "".join(html_parts)


def _process_inline_markdown(text: str) -> str:
    """
    Process inline markdown formatting (bold, italic, code, links).
    
    Args:
        text: Text with inline markdown
        
    Returns:
        HTML string with formatting applied
    """
    if not text:
        return ""
    
    import re
    
    # Escape HTML first
    text = _escape_html(text)
    
    # Inline code: `code` (do this before bold/italic to avoid conflicts)
    text = re.sub(r'`([^`]+)`', r'<code class="mono">\1</code>', text)
    
    # Bold: **text** (but not if it's part of code)
    def replace_bold(match):
        content = match.group(1)
        if '<code' in content or '</code>' in content:
            return match.group(0)  # Don't process if contains code
        return f'<strong>{content}</strong>'
    text = re.sub(r'\*\*([^*]+)\*\*', replace_bold, text)
    
    # Bold: __text__
    text = re.sub(r'__([^_]+)__', r'<strong>\1</strong>', text)
    
    # Italic: *text* (but not if it's part of bold or code)
    def replace_italic(match):
        content = match.group(1)
        if '<code' in content or '</code>' in content or '<strong' in content:
            return match.group(0)
        return f'<em>{content}</em>'
    text = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)', replace_italic, text)
    
    # Links: [text](url)
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2" target="_blank">\1</a>', text)
    
    return text


def format_final_review_markdown(review: dict) -> str:
    """
    Format the final refined review as Markdown for display in gr.Markdown.

    Prefers the pre-built ``review_markdown`` field from the refiner.
    Falls back to assembling markdown from the structured JSON fields
    (summary, strengths, weaknesses, etc.) so that markdown formatting
    and LaTeX equations render correctly via Gradio's native renderer.
    """
    # 1. If the refiner already produced a markdown string, use it directly.
    markdown_text = review.get("review_markdown") or review.get("review") or ""

    # 2. Try structured JSON fields (from review_json or top-level dict)
    review_json = review.get("review_json")
    if review_json:
        if isinstance(review_json, str):
            review_json = safe_json_parse(review_json)
    structured = review_json if (review_json and isinstance(review_json, dict)) else None
    if not structured and _looks_like_raw_json(review):
        structured = review

    if structured:
        parts: list[str] = []

        # Scores bar
        scores = review.get("scores", {}) or {}
        score_fields = [
            ("soundness", "Soundness"),
            ("presentation", "Presentation"),
            ("contribution", "Contribution"),
            ("rating", "Rating"),
            ("confidence", "Confidence"),
            ("decision", "Decision"),
        ]
        score_items = []
        for key, label in score_fields:
            val = scores.get(key) or structured.get(key)
            if val is not None and val != "":
                score_items.append(f"**{label}:** {val}")
        if score_items:
            parts.append(" · ".join(score_items))
            parts.append("")  # blank line

        # Sections
        section_map = [
            ("summary", "Summary"),
            ("strengths", "Strengths"),
            ("weaknesses", "Weaknesses"),
            ("questions", "Questions"),
        ]
        for key, heading in section_map:
            val = structured.get(key)
            if not val:
                continue
            if isinstance(val, list):
                body = "\n".join(f"- {item}" for item in val if item)
            else:
                body = str(val)
            if body.strip():
                parts.append(f"## {heading}\n\n{body}")

        if parts:
            return "\n\n".join(parts)

    # 3. Fall back to raw markdown text
    if markdown_text.strip():
        return markdown_text

    return "*Final review content not available.*"


def format_raw_json(review: dict) -> str:
    """
    Format the complete review as a JSON code block.
    
    Args:
        review: The complete review dictionary
        
    Returns:
        JSON formatted as a Markdown code block
    """
    try:
        return "```json\n" + json.dumps(review, indent=2, ensure_ascii=False) + "\n```"
    except Exception:
        return str(review)
