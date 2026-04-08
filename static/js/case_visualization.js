// ReviewGrounder Case Study Data - ReviewGrounder vs DeepReviewer comparison
// Case studies data is stored here in the caseData object.
// To add new cases: add an entry with keys: title, question, answer, success, failure.

const caseData = {
  1: {
    title: "SEA Paper Review Comparison",
    question: "SEA (Sparse linear Attention with Estimated Attention mask) — A test-time linear-complexity attention module.",
    answer: "Comparison of review substantiveness: evidence-based vs. verbose/superficial",
    success: {
      title: "ReviewGrounder (Evidence-Grounded)",
      steps: [
        {
          title: "Weaknesses (Excerpt)",
          tool: "Evidence-Based",
          subgoal: "Grounded critiques with specific references",
          command: "ReviewGrounder anchors each critique to verifiable paper content",
          result: "<b>Adaptation cost not fully quantified</b> — Section 3.3 describes replacing the teacher's attention with SEA and fine-tuning via KD, but the paper does not report additional training time, data size, or compute needed. (Section 3.3)<br><br><b>CNN decoder architecture details missing</b> — Kernel sizes, strides, padding, and channel counts are omitted, hindering reproducibility. (Section 3.1, CNN Decoder)<br><br><b>Hyper-parameter selection guidance limited</b> — No systematic procedure for choosing K or base k for new tasks. (Table ablation_k, Section 4.3)",
          type: "success"
        },
        {
          title: "Strengths (Excerpt)",
          tool: "Concise & Specific",
          subgoal: "Accurate capture of contributions",
          command: "ReviewGrounder cites exact sections and results",
          result: "Introduces a two-phase linear-attention pipeline with provable O(T) inference cost (Section 3.1, Fig. 1). Proposes FlatCSR achieving 6.6× faster sparse operations than COO (Table 1). Shows k can be increased post-training to improve accuracy (Section 4.3).",
          type: "success"
        }
      ],
      verifier: "Evidence-grounded, actionable, concise",
      solution: "ReviewGrounder produces substantive critiques with specific section/equation/table references, enabling authors to understand and address feedback."
    },
    failure: {
      title: "DeepReviewer-14B (Baseline)",
      steps: [
        {
          title: "Weaknesses (Excerpt)",
          tool: "Generic",
          subgoal: "Lengthy but repetitive",
          command: "DeepReviewer generates long paragraphs",
          result: "Despite the strengths... I have identified several weaknesses. First, the paper lacks a clear explanation of how the method can be applied to pre-trained Transformer models... The paper states that the SEA attention mechanism replaces the original attention mechanism and is trained using knowledge distillation, but it does not provide a clear explanation... [repeats similar points across multiple paragraphs] Second, the paper's experimental evaluation is limited in scope... Third, the paper lacks comprehensive analysis... [continues with verbose, repetitive structure]",
          type: "error"
        },
        {
          title: "Strengths (Excerpt)",
          tool: "Verbose",
          subgoal: "Broad praise",
          command: "DeepReviewer provides lengthy summaries",
          result: "I find several aspects of this paper to be particularly strong. First, the core idea of combining kernel-based attention with a learned sparse mask is novel... The empirical results are compelling... The authors demonstrate comparable or better performance... [repeats similar praises; lacks specific section/table citations]",
          type: "warning"
        }
      ],
      verifier: "Verbose, repetitive, lacks paper-specific grounding",
      solution: "DeepReviewer produces lengthy but superficial feedback that echoes generic critique templates without anchoring to specific paper content."
    }
  }
};

// Initialize case visualization
document.addEventListener('DOMContentLoaded', function() {
  let currentCase = 1;
  let currentView = 'success'; // 'success' = ReviewGrounder, 'failure' = DeepReviewer

  const container = document.getElementById('case-visualization-container');

  // Create example selection buttons (single case for ReviewGrounder)
  const buttonsContainer = document.querySelector('.case-example-buttons');
  if (buttonsContainer) {
    Object.keys(caseData).forEach(caseId => {
      const btn = document.createElement('button');
      btn.className = 'button is-info is-outlined example-button';
      btn.setAttribute('data-case', caseId);
      btn.textContent = caseData[caseId].title;
      btn.addEventListener('click', () => selectCase(caseId));
      buttonsContainer.appendChild(btn);
    });
  }

  function selectCase(caseId) {
    currentCase = caseId;
    document.querySelectorAll('.example-button').forEach(btn => {
      btn.classList.remove('is-active');
      if (btn.getAttribute('data-case') == caseId) {
        btn.classList.add('is-active');
      }
    });
    renderCase();
  }

  function toggleView(view) {
    currentView = view;
    document.querySelectorAll('.case-toggle-btn').forEach(btn => {
      btn.classList.remove('active');
    });
    const activeBtn = document.querySelector(`.case-toggle-btn[data-view="${view}"]`);
    if (activeBtn) activeBtn.classList.add('active');
    renderCase();
  }

  function renderCase() {
    const caseInfo = caseData[currentCase];
    const viewData = currentView === 'success' ? caseInfo.success : caseInfo.failure;
    const isSuccess = currentView === 'success';

    const html = `
      <div class="case-container">
        <div class="case-main-layout">
          <div class="case-sidebar">
            <div class="case-question-box">
              <h3>Paper Under Review</h3>
              <div class="question-text">${caseInfo.question}</div>
              <div class="answer-text">${caseInfo.answer}</div>
            </div>
            <div class="case-toggle-container">
              <button class="case-toggle-btn success-btn ${currentView === 'success' ? 'active' : ''}"
                      data-view="success" onclick="window.caseViz.toggleView('success')">
                ✓ ReviewGrounder
              </button>
              <button class="case-toggle-btn failure-btn ${currentView === 'failure' ? 'active' : ''}"
                      data-view="failure" onclick="window.caseViz.toggleView('failure')">
                ✗ DeepReviewer-14B
              </button>
            </div>
          </div>
          <div class="case-content">
            <div class="case-display-box ${isSuccess ? 'success' : 'failure'}">
              <h4>${viewData.title}</h4>
              ${viewData.steps.map((step, index) => `
                <div class="action-step" data-step-index="${index}">
                  <div class="action-step-header">
                    <div class="action-step-left">
                      <div class="step-number">${index + 1}</div>
                      <h5>${step.title}</h5>
                    </div>
                    <div class="step-toggle-icon">▼</div>
                  </div>
                  <div class="action-step-content">
                    <div class="action-step-right">
                      <div class="step-detail-box">
                        <div class="step-item">
                          <span class="step-label">Style:</span>
                          <span class="tool-name">${step.tool}</span>
                        </div>
                        <div class="step-item">
                          <span class="step-label">Excerpt:</span>
                          <span class="step-content ${
                            step.type === 'error' ? 'result-error' :
                            step.type === 'success' ? 'result-success' : 'result-warning'
                          }">${step.result}</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              `).join('')}
              <div class="verifier-box">
                <strong>Summary:</strong> ${viewData.verifier}
              </div>
              <div class="solution-box">
                <h5>Takeaway</h5>
                <p>${viewData.solution}</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    `;

    if (container) container.innerHTML = html;

    // Accordion
    const stepHeaders = container ? container.querySelectorAll('.action-step-header') : [];
    stepHeaders.forEach((header) => {
      header.addEventListener('click', function() {
        const step = this.closest('.action-step');
        step.classList.toggle('expanded');
      });
    });
    const firstStep = container ? container.querySelector('.action-step') : null;
    if (firstStep) firstStep.classList.add('expanded');
  }

  window.caseViz = { selectCase, toggleView };
  selectCase(1);
});
