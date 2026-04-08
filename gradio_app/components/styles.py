"""
Custom CSS styles for the Review Grounder Gradio app.

Follows the PreScouter Design System:
- Instrument Serif for headlines, DM Sans for body, JetBrains Mono for metadata
- Ink (#0F172A), Background (#FFFFFF), Accent (#1D4ED8), Border (#E2E8F0)
- 2px radius, border-led cards, editorial typography
"""


def get_custom_css() -> str:
    return """
    /* ===== Google Fonts ===== */
    @import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=DM+Sans:ital,wght@0,400;0,500;0,600;0,700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* ===== Design Tokens ===== */
    :root {
        --ink: #0F172A;
        --bg: #FFFFFF;
        --bg-tint: #F8FAFC;
        --accent: #1D4ED8;
        --accent-dark: #1E40AF;
        --border: #E2E8F0;
        --border-strong: #CBD5E1;
        --muted: #475569;
        --light: #94A3B8;
        --radius: 2px;
        --shadow-soft: 0 4px 24px rgba(0,0,0,0.03);
        --shadow-hover: 0 8px 40px rgba(29,78,216,0.08);
        --font-serif: Georgia, serif;
        --font-sans: 'DM Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        --font-mono: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace;
    }

    /* ===== Global Styles ===== */
    .html-container {
        padding: 0 !important;
    }
    .gradio-container {
        max-width: 1200px !important;
        margin: 0 auto !important;
        font-family: var(--font-sans) !important;
        background: var(--bg) !important;
        color: var(--ink) !important;
    }

    .gradio-container .prose {
        font-family: var(--font-sans) !important;
    }

    /* ===== Header ===== */
    .app-header {
        border-bottom: 1px solid var(--border);
        padding: 32px 0 24px 0;
        margin-bottom: 32px;
    }
    .app-header-eyebrow {
        font-family: var(--font-mono);
        font-size: 11px;
        font-weight: 500;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--accent);
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .app-header-eyebrow::before {
        content: '';
        display: inline-block;
        width: 24px;
        height: 1px;
        background: var(--accent);
    }
    .app-header-title {
        font-family: var(--font-serif);
        font-size: 46px;
        font-weight: 400;
        line-height: 1.08;
        color: var(--ink);
        margin: 0 0 8px 0;
        letter-spacing: -0.01em;
    }
    .app-header-subtitle {
        font-family: var(--font-sans);
        font-size: 16px;
        color: var(--muted);
        line-height: 1.6;
        max-width: 600px;
        margin: 0;
    }

    /* ===== Privacy Notice ===== */
    .privacy-notice {
        background: var(--bg-tint);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 12px 16px;
        margin-top: 16px;
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 13px;
        color: var(--muted);
    }
    .privacy-notice strong {
        color: var(--ink);
    }

    /* ===== API Key Section (prominent) ===== */
    .api-key-section {
        background: var(--bg-tint);
        border: 1px solid var(--accent);
        border-left: 3px solid var(--accent);
        border-radius: var(--radius);
        padding: 20px 24px;
        margin-bottom: 24px;
    }
    .api-key-section-title {
        font-family: var(--font-mono);
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--accent);
        margin-bottom: 8px;
    }
    .api-key-section-desc {
        font-size: 14px;
        color: var(--muted);
        margin-bottom: 12px;
        line-height: 1.5;
    }

    /* ===== Panel Cards ===== */
    .panel-card {
        background: var(--bg);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 24px;
        box-shadow: var(--shadow-soft);
    }
    .panel-title {
        font-family: var(--font-serif);
        font-size: 25px;
        font-weight: 400;
        color: var(--ink);
        margin: 0 0 20px 0;
        line-height: 1.18;
    }
    .panel-eyebrow {
        font-family: var(--font-mono);
        font-size: 10px;
        font-weight: 500;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--accent);
        margin-bottom: 6px;
    }

    /* ===== How It Works ===== */
    .how-it-works {
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 16px 20px;
        margin-bottom: 20px;
        background: var(--bg);
    }
    .how-it-works-title {
        font-family: var(--font-mono);
        font-size: 10px;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--muted);
        margin-bottom: 12px;
    }
    .step-item {
        display: flex;
        align-items: flex-start;
        gap: 10px;
        margin-bottom: 8px;
        font-size: 14px;
        color: var(--ink);
        line-height: 1.5;
    }
    .step-number {
        background: var(--accent);
        color: white;
        width: 20px;
        height: 20px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-family: var(--font-mono);
        font-size: 11px;
        font-weight: 600;
        flex-shrink: 0;
        margin-top: 1px;
    }

    /* ===== Upload Area ===== */
    .upload-area {
        border: 1px dashed var(--border-strong) !important;
        border-radius: var(--radius) !important;
        padding: 32px 20px !important;
        text-align: center;
        background: var(--bg-tint) !important;
        transition: border-color 0.3s, background 0.3s;
        margin-bottom: 16px;
    }
    .upload-area:hover {
        border-color: var(--accent) !important;
        background: #F1F5F9 !important;
    }
    .file-upload-minimal .gr-formatted-text,
    .file-upload-minimal .gr-box > div:not([class*="file"]):not([class*="preview"]),
    #pdf-upload .gr-formatted-text,
    #pdf-upload .wrap-inner .gr-formatted-text {
        display: none !important;
    }
    .upload-hint {
        color: var(--light);
        font-size: 13px;
        margin-top: 4px;
        font-family: var(--font-mono);
        font-size: 11px;
        letter-spacing: 0.02em;
    }

    /* ===== Primary Action Button ===== */
    .primary-btn {
        background: var(--accent) !important;
        color: white !important;
        padding: 14px 28px !important;
        border-radius: var(--radius) !important;
        font-family: var(--font-sans) !important;
        font-weight: 600 !important;
        font-size: 15px !important;
        border: none !important;
        cursor: pointer !important;
        transition: background 0.3s, box-shadow 0.3s !important;
        width: 100% !important;
        letter-spacing: 0.01em;
    }
    .primary-btn:hover {
        background: var(--accent-dark) !important;
        box-shadow: var(--shadow-hover) !important;
    }
    .primary-btn:disabled {
        opacity: 0.5 !important;
        cursor: not-allowed !important;
    }

    /* ===== Advanced Settings Accordion ===== */
    .advanced-settings .label-wrap {
        background: var(--bg-tint) !important;
        border-radius: var(--radius) !important;
        padding: 10px 16px !important;
        border: 1px solid var(--border) !important;
    }
    .advanced-settings .label-wrap span {
        font-family: var(--font-sans) !important;
        font-weight: 500 !important;
        font-size: 14px !important;
        color: var(--muted) !important;
    }

    /* ===== Pipeline Progress Section ===== */
    .pipeline-progress {
        border: 1px solid var(--border);
        border-radius: var(--radius);
        background: var(--bg);
        margin-bottom: 24px;
        overflow: hidden;
    }
    .pipeline-progress-header {
        font-family: var(--font-mono);
        font-size: 10px;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--muted);
        padding: 12px 16px;
        border-bottom: 1px solid var(--border);
        background: var(--bg-tint);
    }
    .pipeline-steps {
        display: flex;
        align-items: center;
        padding: 14px 16px;
        gap: 0;
        flex-wrap: nowrap;
        overflow-x: auto;
    }
    .pipeline-step {
        display: flex;
        align-items: center;
        gap: 6px;
        white-space: nowrap;
        font-size: 12px;
        font-family: var(--font-sans);
        color: var(--light);
        font-weight: 500;
    }
    .pipeline-step.active {
        color: var(--accent);
        font-weight: 600;
    }
    .pipeline-step.done {
        color: #059669;
    }
    .pipeline-step-icon {
        font-size: 14px;
    }
    .pipeline-connector {
        width: 24px;
        height: 1px;
        background: var(--border);
        margin: 0 4px;
        flex-shrink: 0;
    }
    .pipeline-connector.done {
        background: #059669;
    }

    /* ===== Terminal Log ===== */
    .terminal-log {
        border: 1px solid var(--border);
        border-radius: var(--radius);
        overflow: hidden;
        margin-bottom: 20px;
    }
    .terminal-log-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 8px 16px;
        background: #0F172A;
        border-bottom: 1px solid #1E293B;
    }
    .terminal-log-title {
        font-family: var(--font-mono);
        font-size: 11px;
        font-weight: 500;
        color: #64748B;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .terminal-log-dots {
        display: flex;
        gap: 6px;
    }
    .terminal-log-dots span {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        display: inline-block;
    }
    .terminal-log-dots .dot-red { background: #EF4444; }
    .terminal-log-dots .dot-yellow { background: #F59E0B; }
    .terminal-log-dots .dot-green { background: #10B981; }

    .terminal-log textarea,
    .status-log textarea {
        font-family: var(--font-mono) !important;
        font-size: 12px !important;
        background: #0F172A !important;
        color: #A5F3FC !important;
        border: none !important;
        border-radius: 0 !important;
        padding: 12px 16px !important;
        line-height: 1.6 !important;
    }

    /* ===== Results Panel ===== */
    .results-panel {
        background: var(--bg);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 24px;
        box-shadow: var(--shadow-soft);
        min-height: 500px;
    }
    .results-placeholder {
        text-align: center !important;
        padding: 64px 24px;
        margin: 0 auto;
    }
    .results-placeholder-icon {
        font-size: 3em;
        margin-bottom: 16px;
        opacity: 0.5;
    }
    .results-placeholder h3 {
        font-family: var(--font-serif);
        font-size: 25px;
        font-weight: 400;
        color: var(--ink);
        margin: 0 0 8px 0;
    }
    .results-placeholder p {
        color: var(--muted);
        font-size: 14px;
        line-height: 1.6;
        max-width: 440px;
        margin: 0 auto;
    }

    /* ===== Tab Styling ===== */
    .tabs .tab-nav {
        border-bottom: 1px solid var(--border) !important;
    }
    .tabs .tab-nav button {
        font-family: var(--font-sans) !important;
        font-weight: 500 !important;
        font-size: 14px !important;
        color: var(--muted) !important;
        padding: 12px 20px !important;
        border: none !important;
        background: transparent !important;
        transition: color 0.3s !important;
    }
    .tabs .tab-nav button.selected {
        color: var(--accent) !important;
        border-bottom: 2px solid var(--accent) !important;
    }
    .tabs .tab-nav button:hover {
        color: var(--ink) !important;
    }

    /* ===== Review Content ===== */
    .review-message {
        color: var(--muted);
        font-style: italic;
        padding: 1em;
    }
    .review-draft-content,
    .initial-draft-card {
        max-width: 100%;
    }
    .review-text {
        white-space: pre-wrap;
        word-break: break-word;
        line-height: 1.7;
        color: var(--ink);
        font-family: var(--font-sans);
    }

    /* Final Review toolbar */
    .final-review-toolbar {
        display: flex;
        justify-content: flex-end;
        margin-bottom: 12px;
    }
    .copy-final-btn {
        padding: 6px 14px;
        border-radius: var(--radius);
        border: 1px solid var(--border);
        background: var(--bg-tint);
        color: var(--muted);
        font-family: var(--font-mono);
        font-size: 11px;
        font-weight: 500;
        letter-spacing: 0.02em;
        cursor: pointer;
        transition: border-color 0.3s, color 0.3s;
    }
    .copy-final-btn:hover {
        border-color: var(--accent);
        color: var(--accent);
    }

    /* Results markdown */
    .results-panel .gr-markdown,
    .results-panel .prose {
        line-height: 1.7 !important;
        color: var(--ink) !important;
        font-family: var(--font-sans) !important;
        max-width: 100% !important;
    }
    .results-panel .gr-markdown h1,
    .results-panel .gr-markdown h2,
    .results-panel .gr-markdown h3 {
        font-family: var(--font-serif) !important;
        margin-top: 1em !important;
        margin-bottom: 0.5em !important;
        color: var(--ink) !important;
        font-weight: 400 !important;
    }
    .results-panel .gr-markdown ul,
    .results-panel .gr-markdown ol {
        padding-left: 1.5em !important;
        margin: 0.5em 0 !important;
    }

    /* ===== Card Grid for Results ===== */
    .card-grid {
        display: flex;
        flex-direction: column;
        gap: 12px;
    }
    .card {
        background: var(--bg);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 16px 20px;
    }
    .card h4 {
        font-family: var(--font-sans);
        font-weight: 600;
        font-size: 15px;
        color: var(--ink);
        margin: 0 0 12px 0;
    }
    .kv {
        margin: 8px 0;
        padding: 12px;
        background: var(--bg-tint);
        border-radius: var(--radius);
        border: 1px solid var(--border);
    }
    .k {
        font-family: var(--font-mono);
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: var(--muted);
        margin-bottom: 6px;
    }
    .v {
        color: var(--ink);
        line-height: 1.65;
        font-size: 14px;
        white-space: pre-wrap;
        word-break: break-word;
    }
    details {
        background: var(--bg);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 10px 12px;
        margin-top: 6px;
    }
    summary {
        cursor: pointer;
        font-weight: 600;
        font-size: 14px;
        color: var(--ink);
    }
    .mono {
        font-family: var(--font-mono);
        font-size: 12px;
    }
    .pill {
        display: inline-block;
        padding: 2px 8px;
        border-radius: var(--radius);
        border: 1px solid var(--border);
        background: var(--bg-tint);
        color: var(--accent);
        font-family: var(--font-mono);
        font-size: 10px;
        font-weight: 500;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-left: 8px;
    }

    /* ===== Related Work Cards ===== */
    .related-work-container {
        font-family: var(--font-sans);
        max-width: 100%;
    }
    .related-work-container h3 {
        font-family: var(--font-serif);
        font-size: 21px;
        font-weight: 400;
        color: var(--ink);
        margin-bottom: 16px;
    }
    .related-paper-card {
        background: var(--bg);
        border: 1px solid var(--border);
        border-left: 3px solid var(--accent);
        border-radius: var(--radius);
        padding: 16px;
        margin-bottom: 12px;
        transition: box-shadow 0.3s;
    }
    .related-paper-card:hover {
        box-shadow: var(--shadow-hover);
    }
    .paper-header {
        font-family: var(--font-sans);
        font-weight: 600;
        font-size: 14px;
        color: var(--ink);
        margin-bottom: 12px;
    }
    .paper-field {
        margin: 6px 0;
        padding: 8px 10px;
        background: var(--bg-tint);
        border-radius: var(--radius);
    }
    .paper-field-label {
        font-family: var(--font-mono);
        font-size: 10px;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: var(--muted);
        margin-bottom: 4px;
    }
    .paper-field-value {
        color: var(--ink);
        font-size: 14px;
        line-height: 1.6;
    }

    /* ===== Collapsible Details (unified) ===== */
    .card-details {
        background: var(--bg);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 10px 12px;
        margin-top: 6px;
    }
    .card-details summary {
        cursor: pointer;
        font-weight: 600;
        font-size: 14px;
        color: var(--ink);
        line-height: 1.5;
    }
    .card-details .details-body {
        margin-top: 8px;
        padding-top: 8px;
        border-top: 1px solid var(--border);
    }
    .card-details .details-body .paper-field-label {
        margin-top: 6px;
    }

    /* ===== Footer ===== */
    .app-footer {
        text-align: center;
        padding: 24px 0;
        color: var(--light);
        font-size: 13px;
        border-top: 1px solid var(--border);
        margin-top: 48px;
        font-family: var(--font-sans);
    }
    .app-footer p {
        margin: 2px 0;
    }

    /* ===== Gradio Overrides ===== */
    .gradio-container input,
    .gradio-container textarea,
    .gradio-container select {
        font-family: var(--font-sans) !important;
        border-radius: var(--radius) !important;
    }
    .gradio-container input:focus,
    .gradio-container textarea:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 1px var(--accent) !important;
    }
    .gradio-container label {
        font-family: var(--font-sans) !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        color: var(--ink) !important;
    }
    .gradio-container .gr-check-radio {
        font-family: var(--font-sans) !important;
    }
    .gradio-container .gr-input-label {
        font-family: var(--font-sans) !important;
    }
    /* Hide Gradio footer */
    footer {
        display: none !important;
    }

    /* =============================================================
       DARK MODE
       Triggered by OS / browser preference or Gradio's dark toggle.
       Override CSS custom properties so all components adapt.
       ============================================================= */
    @media (prefers-color-scheme: dark) {
        :root {
            --ink: #E2E8F0;
            --bg: #0F172A;
            --bg-tint: #1E293B;
            --accent: #60A5FA;
            --accent-dark: #93C5FD;
            --border: #334155;
            --border-strong: #475569;
            --muted: #94A3B8;
            --light: #64748B;
            --shadow-soft: 0 4px 24px rgba(0,0,0,0.2);
            --shadow-hover: 0 8px 40px rgba(96,165,250,0.1);
        }
    }
    /* Gradio adds .dark class on the body/container when toggled */
    .dark {
        --ink: #E2E8F0;
        --bg: #0F172A;
        --bg-tint: #1E293B;
        --accent: #60A5FA;
        --accent-dark: #93C5FD;
        --border: #334155;
        --border-strong: #475569;
        --muted: #94A3B8;
        --light: #64748B;
        --shadow-soft: 0 4px 24px rgba(0,0,0,0.2);
        --shadow-hover: 0 8px 40px rgba(96,165,250,0.1);
    }
    
    /* -- Dark mode surface overrides -- */
    @media (prefers-color-scheme: dark) {
        .gradio-container {
            background: var(--bg) !important;
            color: var(--ink) !important;
        }
        .panel-card,
        .results-panel {
            background: var(--bg) !important;
            border-color: var(--border) !important;
        }
        .related-paper-card {
            background: var(--bg-tint) !important;
            border-color: var(--border) !important;
        }
        .paper-field,
        .kv {
            background: #0F172A !important;
            border-color: var(--border) !important;
        }
        .card-details,
        details {
            background: var(--bg-tint) !important;
            border-color: var(--border) !important;
        }
        .card {
            background: var(--bg-tint) !important;
            border-color: var(--border) !important;
        }
        .how-it-works {
            background: var(--bg-tint) !important;
            border-color: var(--border) !important;
        }
        .api-key-section {
            background: var(--bg-tint) !important;
            border-color: var(--accent) !important;
        }
        .privacy-notice {
            background: var(--bg-tint) !important;
            border-color: var(--border) !important;
        }
        .upload-area {
            background: var(--bg-tint) !important;
            border-color: var(--border-strong) !important;
        }
        .upload-area:hover {
            border-color: var(--accent) !important;
            background: #1E293B !important;
        }
        .advanced-settings .label-wrap {
            background: var(--bg-tint) !important;
            border-color: var(--border) !important;
        }
        .pill {
            background: var(--bg-tint) !important;
            border-color: var(--border) !important;
        }
        .copy-final-btn {
            background: var(--bg-tint) !important;
            border-color: var(--border) !important;
            color: var(--muted) !important;
        }
        .copy-final-btn:hover {
            border-color: var(--accent) !important;
            color: var(--accent) !important;
        }
        .pipeline-progress {
            background: var(--bg-tint) !important;
            border-color: var(--border) !important;
        }
        .pipeline-progress-header {
            background: #0F172A !important;
            border-color: var(--border) !important;
        }
        .app-header {
            border-color: var(--border) !important;
        }
        .app-footer {
            border-color: var(--border) !important;
        }
        /* Gradio component overrides */
        .gradio-container .gr-block,
        .gradio-container .gr-group,
        .gradio-container .gr-box,
        .gradio-container .block,
        .gradio-container .wrap,
        .gradio-container .gr-panel {
            background: var(--bg) !important;
            border-color: var(--border) !important;
        }
        .gradio-container .gr-accordion,
        .gradio-container .gr-accordion > .block,
        .gradio-container .gr-accordion > .wrap,
        .advanced-settings,
        .advanced-settings .block,
        .advanced-settings .wrap,
        .advanced-settings .gr-group {
            background: var(--bg) !important;
            border-color: var(--border) !important;
        }
        .gradio-container .gr-input,
        .gradio-container .gr-text-input,
        .gradio-container .gr-textbox,
        .gradio-container input,
        .gradio-container textarea,
        .gradio-container select {
            background: var(--bg-tint) !important;
            color: var(--ink) !important;
            border-color: var(--border) !important;
        }
        .gradio-container label,
        .gradio-container .gr-input-label {
            color: var(--ink) !important;
        }
        .gradio-container .gr-form > .wrap,
        .gradio-container .gr-form .block {
            background: var(--bg) !important;
        }
        .tabs .tab-nav {
            border-color: var(--border) !important;
        }
        .tabs .tab-nav button {
            color: var(--muted) !important;
        }
        .tabs .tab-nav button.selected {
            color: var(--accent) !important;
            border-color: var(--accent) !important;
        }
        .tabs .tab-nav button:hover {
            color: var(--ink) !important;
        }
        .results-panel .gr-markdown,
        .results-panel .prose {
            color: var(--ink) !important;
        }
        .results-panel .gr-markdown h1,
        .results-panel .gr-markdown h2,
        .results-panel .gr-markdown h3 {
            color: var(--ink) !important;
        }
        .results-placeholder h3 {
            color: var(--ink) !important;
        }
        .results-placeholder p {
            color: var(--muted) !important;
        }
        .wrap {
            background: none !important;
        }
    }

    /* Duplicate for Gradio .dark class toggle */
    .dark .gradio-container {
        background: var(--bg) !important;
        color: var(--ink) !important;
    }
    .dark .panel-card,
    .dark .results-panel {
        background: var(--bg) !important;
        border-color: var(--border) !important;
    }
    .dark .related-paper-card {
        background: var(--bg-tint) !important;
        border-color: var(--border) !important;
    }
    .dark .paper-field,
    .dark .kv {
        background: #0F172A !important;
        border-color: var(--border) !important;
    }
    .dark .card-details,
    .dark details {
        background: var(--bg-tint) !important;
        border-color: var(--border) !important;
    }
    .dark .card {
        background: var(--bg-tint) !important;
        border-color: var(--border) !important;
    }
    .dark .how-it-works {
        background: var(--bg-tint) !important;
        border-color: var(--border) !important;
    }
    .dark .api-key-section {
        background: var(--bg-tint) !important;
        border-color: var(--accent) !important;
    }
    .dark .privacy-notice {
        background: var(--bg-tint) !important;
        border-color: var(--border) !important;
    }
    .dark .upload-area {
        background: var(--bg-tint) !important;
        border-color: var(--border-strong) !important;
    }
    .dark .upload-area:hover {
        border-color: var(--accent) !important;
        background: #1E293B !important;
    }
    .dark .advanced-settings .label-wrap {
        background: var(--bg-tint) !important;
        border-color: var(--border) !important;
    }
    .dark .pill {
        background: var(--bg-tint) !important;
        border-color: var(--border) !important;
    }
    .dark .copy-final-btn {
        background: var(--bg-tint) !important;
        border-color: var(--border) !important;
        color: var(--muted) !important;
    }
    .dark .copy-final-btn:hover {
        border-color: var(--accent) !important;
        color: var(--accent) !important;
    }
    .dark .pipeline-progress {
        background: var(--bg-tint) !important;
        border-color: var(--border) !important;
    }
    .dark .pipeline-progress-header {
        background: #0F172A !important;
        border-color: var(--border) !important;
    }
    .dark .app-header {
        border-color: var(--border) !important;
    }
    .dark .app-footer {
        border-color: var(--border) !important;
    }
    .dark .tabs .tab-nav {
        border-color: var(--border) !important;
    }
    .dark .tabs .tab-nav button {
        color: var(--muted) !important;
    }
    .dark .tabs .tab-nav button.selected {
        color: var(--accent) !important;
        border-color: var(--accent) !important;
    }
    .dark .tabs .tab-nav button:hover {
        color: var(--ink) !important;
    }
    .dark .results-panel .gr-markdown,
    .dark .results-panel .prose {
        color: var(--ink) !important;
    }
    .dark .results-panel .gr-markdown h1,
    .dark .results-panel .gr-markdown h2,
    .dark .results-panel .gr-markdown h3 {
        color: var(--ink) !important;
    }
    .dark .results-placeholder h3 {
        color: var(--ink) !important;
    }
    .dark .results-placeholder p {
        color: var(--muted) !important;
    }

    /*
     * Gradio internal component backgrounds.
     * Gradio wraps every component in .gr-block / .block / .wrap
     * divs that carry their own background-color in dark mode
     * (typically a gray like #374151). We force them to our palette.
     */
    @media (prefers-color-scheme: dark) {
        .gradio-container .gr-block,
        .gradio-container .gr-group,
        .gradio-container .gr-box,
        .gradio-container .block,
        .gradio-container .div {
            border: #314154 !important;
        },
        .gradio-container .wrap {
            background: none !important;
        },
        .gradio-container .gr-panel {
            background: var(--bg) !important;
            border-color: var(--border) !important;
        }
        /* Accordion wrapper and inner content */
        .gradio-container .gr-accordion,
        .gradio-container .gr-accordion > .block,
        .gradio-container .gr-accordion > .wrap,
        .advanced-settings,
        .advanced-settings .block,
        .advanced-settings .wrap,
        .advanced-settings .gr-group {
            background: var(--bg) !important;
            border-color: var(--border) !important;
        }
        /* Input/textarea wrappers */
        .gradio-container .gr-input,
        .gradio-container .gr-text-input,
        .gradio-container .gr-textbox,
        .gradio-container input,
        .gradio-container textarea,
        .gradio-container select {
            background: var(--bg-tint) !important;
            color: var(--ink) !important;
            border-color: var(--border) !important;
        }
        .gradio-container label,
        .gradio-container .gr-input-label {
            color: var(--ink) !important;
        }
        /* Info text below inputs */
        .gradio-container .gr-form > .wrap,
        .gradio-container .gr-form .block {
            background: var(--bg) !important;
        }
        
    }

    .dark .gradio-container .gr-block,
    .dark .gradio-container .gr-group,
    .dark .gradio-container .gr-box,
    .dark .gradio-container .block,
    .dark .gradio-container .wrap,
    .dark .gradio-container .gr-panel {
        background: var(--bg) !important;
        border-color: var(--border) !important;
    }
    .dark .gradio-container .gr-accordion,
    .dark .gradio-container .gr-accordion > .block,
    .dark .gradio-container .gr-accordion > .wrap,
    .dark .advanced-settings,
    .dark .advanced-settings .block,
    .dark .advanced-settings .wrap,
    .dark .advanced-settings .gr-group {
        background: var(--bg) !important;
        border-color: var(--border) !important;
    }
    .dark .gradio-container .gr-input,
    .dark .gradio-container .gr-text-input,
    .dark .gradio-container .gr-textbox,
    .dark .gradio-container input,
    .dark .gradio-container textarea,
    .dark .gradio-container select {
        background: var(--bg-tint) !important;
        color: var(--ink) !important;
        border-color: var(--border) !important;
    }
    .dark .gradio-container label,
    .dark .gradio-container .gr-input-label {
        color: var(--ink) !important;
    }
    .dark .gradio-container .gr-form > .wrap,
    .dark .gradio-container .gr-form .block {
        background: var(--bg) !important;
    }

    """
