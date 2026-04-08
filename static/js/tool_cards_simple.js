// Simplified Tool Cards Display

let selectedToolId = null;

document.addEventListener('DOMContentLoaded', function() {
  renderToolSidebar();
  // Select first tool by default
  if (toolsData && toolsData.length > 0) {
    selectTool(toolsData[0].id);
  }
});

function renderToolSidebar() {
  const sidebar = document.getElementById('tools-sidebar');
  if (!sidebar || typeof toolsData === 'undefined') return;

  sidebar.innerHTML = toolsData.map(tool => `
    <button class="tool-selector-btn" onclick="window.toolCards.selectTool('${tool.id}')" data-tool-id="${tool.id}">
      <span class="tool-selector-icon">${tool.icon}</span>
      <span>${tool.name}</span>
    </button>
  `).join('');
}

function selectTool(toolId) {
  selectedToolId = toolId;

  // Update active button
  document.querySelectorAll('.tool-selector-btn').forEach(btn => {
    btn.classList.remove('active');
  });
  const activeBtn = document.querySelector(`[data-tool-id="${toolId}"]`);
  if (activeBtn) {
    activeBtn.classList.add('active');
  }

  // Render selected tool card
  renderToolCard(toolId);
}

function renderToolCard(toolId) {
  const container = document.getElementById('tools-grid');
  if (!container || typeof toolsData === 'undefined') return;

  const tool = toolsData.find(t => t.id === toolId);
  if (!tool) return;

  container.innerHTML = createToolCard(tool);
}

function createToolCard(tool) {
  // Create demo command HTML (show only first one)
  const demoHTML = tool.demoCommands && tool.demoCommands.length > 0 ? `
    <div class="tool-card-section">
      <div class="tool-section-label">Example Usage</div>
      <div class="tool-code-example"><code>${escapeHtml(tool.demoCommands[0].command)}</code></div>
      <div class="tool-code-desc">${tool.demoCommands[0].description}</div>
    </div>
  ` : '';

  return `
    <div class="tool-card">
      <div class="tool-card-header">
        <h3 class="tool-card-title">
          <span class="tool-icon">${tool.icon}</span>
          ${tool.name}
        </h3>
      </div>

      <p class="tool-card-description">${tool.description}</p>

      <div class="tool-card-section">
        <div class="tool-section-label">Inputs</div>
        <div class="tool-inputs">
          ${tool.inputs.map(input => `
            <div class="tool-param">
              <span class="tool-param-name">${input.name}</span>
              <span class="tool-param-type">${input.type}</span> - ${input.desc}
            </div>
          `).join('')}
        </div>
      </div>

      <div class="tool-card-section">
        <div class="tool-section-label">Output</div>
        <div class="tool-output">
          <div class="tool-param">
            <span class="tool-param-type">${tool.output.type}</span> - ${tool.output.desc}
          </div>
        </div>
      </div>

      ${demoHTML}

      <div class="tool-card-footer">
        <button class="tool-link tool-metadata-btn" onclick="window.toolCards.showMetadata('${tool.id}')">
          <i class="fas fa-info-circle"></i>
          Metadata
        </button>
        ${tool.codeLink && tool.codeLink !== '#' ? `<a href="${tool.codeLink}" class="tool-link" target="_blank"><i class="fab fa-github"></i> Code</a>` : ''}
      </div>
    </div>
  `;
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

function showMetadata(toolId) {
  const tool = toolsData.find(t => t.id === toolId);
  if (!tool) return;

  const modalHTML = `
    <div class="tool-metadata-modal" onclick="window.toolCards.closeMetadata(event)">
      <div class="tool-metadata-content" onclick="event.stopPropagation()">
        <button class="tool-metadata-close" onclick="window.toolCards.closeMetadata()">
          <i class="fas fa-times"></i>
        </button>

        <div class="tool-metadata-header">
          <span class="tool-icon-large">${tool.icon}</span>
          <h2>${tool.name}</h2>
        </div>

        <div class="tool-metadata-body">
          <section class="metadata-section">
            <h3>Description</h3>
            <p>${tool.description}</p>
          </section>

          <section class="metadata-section">
            <h3>Inputs</h3>
            ${tool.inputs.map(input => `
              <div class="metadata-param">
                <strong>${input.name}</strong> <em>(${input.type})</em> - ${input.desc}
              </div>
            `).join('')}
          </section>

          <section class="metadata-section">
            <h3>Output</h3>
            <div class="metadata-param">
              <em>(${tool.output.type})</em> - ${tool.output.desc}
            </div>
          </section>

          ${tool.demoCommands && tool.demoCommands.length > 0 ? `
            <section class="metadata-section">
              <h3>Demo Commands</h3>
              ${tool.demoCommands.map(cmd => `
                <div class="metadata-demo">
                  <div class="tool-code-example"><code>${escapeHtml(cmd.command)}</code></div>
                  <p class="tool-code-desc">${cmd.description}</p>
                </div>
              `).join('')}
            </section>
          ` : ''}

          ${tool.limitations && tool.limitations.length > 0 ? `
            <section class="metadata-section">
              <h3>⚠️ Limitations</h3>
              <ul class="metadata-list">
                ${tool.limitations.map(limit => `<li>${limit}</li>`).join('')}
              </ul>
            </section>
          ` : ''}

          ${tool.bestPractices && tool.bestPractices.length > 0 ? `
            <section class="metadata-section">
              <h3>✓ Best Practices</h3>
              <ul class="metadata-list">
                ${tool.bestPractices.map(practice => `<li>${practice}</li>`).join('')}
              </ul>
            </section>
          ` : ''}

          <section class="metadata-section">
            <h3>LLM Engine Required</h3>
            <p><strong>${tool.llmRequired ? 'Yes' : 'No'}</strong></p>
          </section>
        </div>
      </div>
    </div>
  `;

  document.body.insertAdjacentHTML('beforeend', modalHTML);
  document.body.style.overflow = 'hidden';
}

function closeMetadata(event) {
  if (event && event.target.classList.contains('tool-metadata-content')) {
    return;
  }
  const modal = document.querySelector('.tool-metadata-modal');
  if (modal) {
    modal.remove();
    document.body.style.overflow = '';
  }
}

// Expose functions globally
window.toolCards = {
  selectTool,
  showMetadata,
  closeMetadata
};
