// Add copy button to all MathJax formulas
document.addEventListener('DOMContentLoaded', function() {
  // Wait for MathJax to finish rendering
  if (window.MathJax) {
    MathJax.startup.promise.then(() => {
      addCopyButtonsToFormulas();
    });
  }
});

function addCopyButtonsToFormulas() {
  // Find all display math elements ($$...$$)
  const mathElements = document.querySelectorAll('mjx-container[display="true"]');

  mathElements.forEach((mathElement) => {
    // Create wrapper div if not already wrapped
    if (!mathElement.parentElement.classList.contains('formula-wrapper')) {
      const wrapper = document.createElement('div');
      wrapper.className = 'formula-wrapper';
      mathElement.parentNode.insertBefore(wrapper, mathElement);
      wrapper.appendChild(mathElement);

      // Create copy button
      const copyButton = document.createElement('button');
      copyButton.className = 'formula-copy-btn';
      copyButton.innerHTML = '<i class="fas fa-copy"></i>';
      copyButton.title = 'Copy formula';

      // Add click event
      copyButton.addEventListener('click', function(e) {
        e.preventDefault();
        copyFormulaToClipboard(mathElement, copyButton);
      });

      wrapper.appendChild(copyButton);
    }
  });
}

function copyFormulaToClipboard(mathElement, button) {
  // Get the LaTeX source from the MathJax element
  let latexSource = '';

  // Try to get from script tag (MathJax stores original TeX here)
  const scriptTag = mathElement.querySelector('script[type="math/tex"]');
  if (scriptTag) {
    latexSource = scriptTag.textContent;
  } else {
    // Fallback: try to get from data attribute or text content
    latexSource = mathElement.getAttribute('data-latex') || mathElement.textContent;
  }

  // If still empty, try to extract from the original source
  if (!latexSource || latexSource.trim() === '') {
    // Find the nearest .math-scrollable parent or get text content
    const parentScrollable = mathElement.closest('.math-scrollable');
    if (parentScrollable) {
      // Try to find the original TeX in a comment or previous sibling
      const prevSibling = parentScrollable.previousElementSibling;
      if (prevSibling && prevSibling.nodeType === Node.COMMENT_NODE) {
        latexSource = prevSibling.textContent;
      }
    }

    // Last resort: get the MathML or text representation
    if (!latexSource || latexSource.trim() === '') {
      const mathml = mathElement.querySelector('mjx-math');
      if (mathml) {
        latexSource = mathml.getAttribute('alttext') || mathElement.textContent;
      }
    }
  }

  // Copy to clipboard
  navigator.clipboard.writeText(latexSource.trim()).then(() => {
    // Show success feedback
    const originalHTML = button.innerHTML;
    button.innerHTML = '<i class="fas fa-check"></i>';
    button.classList.add('copied');

    setTimeout(() => {
      button.innerHTML = originalHTML;
      button.classList.remove('copied');
    }, 2000);
  }).catch(err => {
    console.error('Failed to copy formula: ', err);
    // Fallback: show error
    button.innerHTML = '<i class="fas fa-times"></i>';
    setTimeout(() => {
      button.innerHTML = '<i class="fas fa-copy"></i>';
    }, 2000);
  });
}

// Re-run after MathJax re-renders (if content changes dynamically)
if (window.MathJax) {
  MathJax.startup.promise.then(() => {
    MathJax.typesetPromise = new Proxy(MathJax.typesetPromise, {
      apply: function(target, thisArg, argumentsList) {
        return Reflect.apply(target, thisArg, argumentsList).then(() => {
          setTimeout(addCopyButtonsToFormulas, 100);
        });
      }
    });
  });
}
