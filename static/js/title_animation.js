// ReviewGrounder Title Hover Animation
document.addEventListener('DOMContentLoaded', function() {
  const agentText = document.getElementById('agent-text');
  const flowText = document.getElementById('flow-text');

  // Agent icons (智能体相关)
  const agentIcons = ['🤖', '⚡', '🎯', '🔮'];

  let agentHoverTimeout = null;
  let flowHoverTimeout = null;

  function createFloatingIcon(icon, element) {
    const rect = element.getBoundingClientRect();
    const iconElement = document.createElement('div');
    iconElement.className = 'float-icon';
    iconElement.textContent = icon;

    // Random horizontal offset: center, or slightly left/right (within ±40px)
    const randomOffsetX = (Math.random() - 0.5) * 80; // -40px to +40px
    iconElement.style.left = `${rect.left + rect.width / 2 + randomOffsetX}px`;
    iconElement.style.top = `${rect.top + window.scrollY}px`;

    document.body.appendChild(iconElement);

    // Remove element after animation
    setTimeout(() => {
      iconElement.remove();
    }, 1500);
  }

  function createFlowWave(element) {
    const rect = element.getBoundingClientRect();

    // Create SVG for curved lines
    const svgNS = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(svgNS, "svg");
    svg.setAttribute("class", "flow-wave-svg");
    svg.style.position = "absolute";
    svg.style.left = `${rect.left}px`;
    svg.style.top = `${rect.top - 20 + window.scrollY}px`;
    svg.style.width = `${rect.width}px`;
    svg.style.height = "60px";
    svg.style.pointerEvents = "none";
    svg.style.overflow = "visible";
    svg.style.zIndex = "100";

    // Create 3 curved paths with different positions
    const curves = [
      { yOffset: 20, amplitude: 8, id: 'curve1' },
      { yOffset: 30, amplitude: 10, id: 'curve2' },
      { yOffset: 40, amplitude: 6, id: 'curve3' }
    ];

    curves.forEach((curve, index) => {
      const path = document.createElementNS(svgNS, "path");

      // Random length between 50% and 100% of the full width
      const randomLength = 0.5 + Math.random() * 0.5; // 0.5 to 1.0
      const width = rect.width;
      const actualWidth = width * randomLength;

      const startY = curve.yOffset;
      const cp1X = actualWidth * 0.25;
      const cp1Y = startY - curve.amplitude;
      const cp2X = actualWidth * 0.75;
      const cp2Y = startY + curve.amplitude;
      const endY = startY;

      const d = `M 0 ${startY} C ${cp1X} ${cp1Y}, ${cp2X} ${cp2Y}, ${actualWidth} ${endY}`;

      path.setAttribute("d", d);
      path.setAttribute("fill", "none");
      path.setAttribute("stroke", "url(#flowGradient)");
      path.setAttribute("stroke-width", "3");
      path.setAttribute("stroke-linecap", "round");
      path.setAttribute("class", `flow-curve flow-curve-${index + 1}`);
      path.style.filter = "drop-shadow(0 0 6px rgba(36, 108, 174, 0.6))";

      // Duration based on length: longer curves take more time (2.5s to 3.5s)
      const duration = 2.5 + randomLength * 1.0; // 2.5s for 50% length, 3.5s for 100% length
      path.style.animationDuration = `${duration}s`;

      svg.appendChild(path);
    });

    // Create gradient definition
    const defs = document.createElementNS(svgNS, "defs");
    const gradient = document.createElementNS(svgNS, "linearGradient");
    gradient.setAttribute("id", "flowGradient");
    gradient.setAttribute("x1", "0%");
    gradient.setAttribute("y1", "0%");
    gradient.setAttribute("x2", "100%");
    gradient.setAttribute("y2", "0%");

    const stops = [
      { offset: "0%", color: "rgba(36, 108, 174, 0)" },
      { offset: "20%", color: "rgba(36, 108, 174, 1)" },
      { offset: "50%", color: "rgba(36, 108, 174, 1)" },
      { offset: "80%", color: "rgba(36, 108, 174, 0.5)" },
      { offset: "100%", color: "rgba(36, 108, 174, 0)" }
    ];

    stops.forEach(stop => {
      const stopElement = document.createElementNS(svgNS, "stop");
      stopElement.setAttribute("offset", stop.offset);
      stopElement.setAttribute("stop-color", stop.color);
      gradient.appendChild(stopElement);
    });

    defs.appendChild(gradient);
    svg.appendChild(defs);

    document.body.appendChild(svg);

    // Remove element after animation (max duration is 3.5s)
    setTimeout(() => {
      svg.remove();
    }, 3500);
  }

  function showAgentIcons(element) {
    const randomIcon = agentIcons[Math.floor(Math.random() * agentIcons.length)];
    createFloatingIcon(randomIcon, element);
  }

  // Hover event for "Agent"
  if (agentText) {
    agentText.addEventListener('mouseenter', function() {
      // Show first icon immediately
      showAgentIcons(agentText);

      // Then show another after delay
      agentHoverTimeout = setTimeout(() => {
        showAgentIcons(agentText);
      }, 800);
    });

    agentText.addEventListener('mouseleave', function() {
      if (agentHoverTimeout) {
        clearTimeout(agentHoverTimeout);
      }
    });
  }

  // Hover event for "Flow"
  if (flowText) {
    flowText.addEventListener('mouseenter', function() {
      // Create wave immediately
      createFlowWave(flowText);

      // Create another wave with delay
      flowHoverTimeout = setTimeout(() => {
        createFlowWave(flowText);
      }, 1000);
    });

    flowText.addEventListener('mouseleave', function() {
      if (flowHoverTimeout) {
        clearTimeout(flowHoverTimeout);
      }
    });
  }
});
