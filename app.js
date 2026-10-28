/**
 * FEDERATED CV QUALITY CONTROL — INTERACTIVE ENGINE
 * Edge Vision • Flower Federated Learning • Differential Privacy Accountant
 */

(function () {
  'use strict';

  // =========================================================================
  // 1. DATA REPOSITORIES & FACTORY CONFIGURATIONS
  // =========================================================================

  const FACTORY_DATA = {
    factory_a: {
      id: 'factory_a',
      name: 'Factory A (Metal Nuts)',
      domain: 'Automotive & Mechanical Fasteners',
      model: 'YOLOv8n-cls (Detroit Edge Node)',
      samples: [
        {
          id: 'a_bent',
          name: 'Bent Flange',
          type: 'defect',
          image: 'samples/factory_a_bent.jpg',
          defectProb: 97.42,
          goodProb: 2.58,
          anomaly: 'Bent Contour / Flange Warp',
          actuator: 'REJECT BIN 2 (SCRAP)',
          latency: '13.8 ms',
          cam: { x: 140, y: 260, r: 70, intensity: 0.95 }
        },
        {
          id: 'a_scratch',
          name: 'Surface Scratch',
          type: 'defect',
          image: 'samples/factory_a_scratch.jpg',
          defectProb: 96.15,
          goodProb: 3.85,
          anomaly: 'Tooling Surface Scratch',
          actuator: 'REJECT BIN 2 (SCRAP)',
          latency: '14.1 ms',
          cam: { x: 215, y: 160, r: 60, intensity: 0.92 }
        },
        {
          id: 'a_color',
          name: 'Oxidation Stain',
          type: 'defect',
          image: 'samples/factory_a_color.jpg',
          defectProb: 94.80,
          goodProb: 5.20,
          anomaly: 'Chemical Discoloration',
          actuator: 'REJECT BIN 2 (SCRAP)',
          latency: '13.5 ms',
          cam: { x: 190, y: 200, r: 75, intensity: 0.88 }
        },
        {
          id: 'a_good',
          name: 'Nominal Nut',
          type: 'good',
          image: 'samples/factory_a_good.jpg',
          defectProb: 0.88,
          goodProb: 99.12,
          anomaly: 'None (Meets ISO-4032)',
          actuator: 'PASS TO PACKAGING',
          latency: '12.9 ms',
          cam: null
        }
      ]
    },
    factory_b: {
      id: 'factory_b',
      name: 'Factory B (Transistors)',
      domain: 'Semiconductor Leadframes & IC Packaging',
      model: 'YOLOv8n-cls (Shenzhen Edge Node)',
      samples: [
        {
          id: 'b_bent_lead',
          name: 'Bent Lead Pin',
          type: 'defect',
          image: 'samples/factory_b_bent_lead.jpg',
          defectProb: 98.65,
          goodProb: 1.35,
          anomaly: 'Lead 3 Coplanarity Deviation',
          actuator: 'REJECT BIN 1 (REWORK)',
          latency: '14.4 ms',
          cam: { x: 280, y: 240, r: 65, intensity: 0.96 }
        },
        {
          id: 'b_damaged',
          name: 'Cracked Package',
          type: 'defect',
          image: 'samples/factory_b_damaged.jpg',
          defectProb: 99.20,
          goodProb: 0.80,
          anomaly: 'Epoxy Case Fracture',
          actuator: 'REJECT BIN 1 (SCRAP)',
          latency: '13.9 ms',
          cam: { x: 190, y: 170, r: 70, intensity: 0.98 }
        },
        {
          id: 'b_cut_lead',
          name: 'Severed Lead',
          type: 'defect',
          image: 'samples/factory_b_cut_lead.jpg',
          defectProb: 97.80,
          goodProb: 2.20,
          anomaly: 'Missing Terminal Pin',
          actuator: 'REJECT BIN 1 (SCRAP)',
          latency: '14.2 ms',
          cam: { x: 120, y: 260, r: 60, intensity: 0.94 }
        },
        {
          id: 'b_good',
          name: 'Pristine TO-92',
          type: 'good',
          image: 'samples/factory_b_good.jpg',
          defectProb: 0.55,
          goodProb: 99.45,
          anomaly: 'None (JEDEC Compliant)',
          actuator: 'PASS TO TAPE-AND-REEL',
          latency: '12.5 ms',
          cam: null
        }
      ]
    },
    factory_c: {
      id: 'factory_c',
      name: 'Factory C (Carpet & Textiles)',
      domain: 'Industrial Weave & Tufted Fabrics',
      model: 'YOLOv8n-cls (Bengaluru Edge Node)',
      samples: [
        {
          id: 'c_hole',
          name: 'Puncture Hole',
          type: 'defect',
          image: 'samples/factory_c_hole.jpg',
          defectProb: 98.90,
          goodProb: 1.10,
          anomaly: 'Weave Rupture / Perforation',
          actuator: 'AUTO-SHEAR & DOCK',
          latency: '15.1 ms',
          cam: { x: 200, y: 190, r: 85, intensity: 0.97 }
        },
        {
          id: 'c_cut',
          name: 'Transverse Cut',
          type: 'defect',
          image: 'samples/factory_c_cut.jpg',
          defectProb: 96.75,
          goodProb: 3.25,
          anomaly: 'Knife Slit in Pile',
          actuator: 'AUTO-SHEAR & DOCK',
          latency: '14.8 ms',
          cam: { x: 170, y: 220, r: 75, intensity: 0.93 }
        },
        {
          id: 'c_thread',
          name: 'Loose Thread',
          type: 'defect',
          image: 'samples/factory_c_thread.jpg',
          defectProb: 95.30,
          goodProb: 4.70,
          anomaly: 'Unlooped Warp Filament',
          actuator: 'FLAG FOR MANUAL TRIM',
          latency: '14.6 ms',
          cam: { x: 230, y: 160, r: 68, intensity: 0.89 }
        },
        {
          id: 'c_good',
          name: 'Uniform Fabric',
          type: 'good',
          image: 'samples/factory_c_good.jpg',
          defectProb: 1.15,
          goodProb: 98.85,
          anomaly: 'None (Density Nominal)',
          actuator: 'WIND TO FINISHED ROLL',
          latency: '13.2 ms',
          cam: null
        }
      ]
    }
  };

  // Real log from server/logs/training_log.json
  const TRAINING_ROUNDS_DATA = [
    { round: 1, clients: 3, epsilon: 0.5, failures: 0, annotation: "Cold-start FedAvg round. Global weights broadcast to 3 factory edge clients." },
    { round: 2, clients: 3, epsilon: 1.0, failures: 0, annotation: "First gradient clipping pass (L2 norm <= 1.0). DP Gaussian noise injected at edge." },
    { round: 3, clients: 3, epsilon: 1.5, failures: 0, annotation: "Model weights beginning inter-factory feature transfer across dissimilar defect domains." },
    { round: 4, clients: 3, epsilon: 2.0, failures: 0, annotation: "Fastener and leadframe boundary features stabilizing in shared representation space." },
    { round: 5, clients: 3, epsilon: 2.5, failures: 0, annotation: "Midpoint validation. Client loss stabilizes across non-IID local partitions." },
    { round: 6, clients: 3, epsilon: 3.0, failures: 0, annotation: "Differential privacy cumulative epsilon reaches 3.0. Zero client drops." },
    { round: 7, clients: 3, epsilon: 3.5, failures: 0, annotation: "High-gradient anomaly patterns generalized into global classification head." },
    { round: 8, clients: 3, epsilon: 4.0, failures: 0, annotation: "Differential privacy noise-to-signal ratio optimal for cross-domain recall." },
    { round: 9, clients: 3, epsilon: 4.5, failures: 0, annotation: "Final fine-tuning iterations. Learning rate decayed to 0.0001." },
    { round: 10, clients: 3, epsilon: 5.0, failures: 0, annotation: "Final global model converged. Total epsilon spent = 5.0 (bounded by delta = 1e-5)." }
  ];

  // =========================================================================
  // 2. STATE MANAGEMENT
  // =========================================================================

  const state = {
    activeFactory: 'factory_a',
    activeSample: null,
    sensitivityThreshold: 50, // %
    camEnabled: true,
    isScanning: false,
    selectedRound: 10,
    isSimulatingRound: false
  };

  // DOM Elements Cache
  const elements = {
    inspectionImage: document.getElementById('inspection-image'),
    heatmapCanvas: document.getElementById('heatmap-overlay-canvas'),
    canvasContainer: document.getElementById('canvas-container'),
    btnToggleCam: document.getElementById('btn-toggle-cam'),
    sampleThumbnailTray: document.getElementById('sample-thumbnail-tray'),
    sampleCountLabel: document.getElementById('sample-count-label'),
    uploadDropzone: document.getElementById('upload-dropzone'),
    fileInput: document.getElementById('file-input'),
    scanStateOverlay: document.getElementById('scan-state-overlay'),

    // HUD Elements
    hudModelTag: document.getElementById('hud-model-tag'),
    verdictBanner: document.getElementById('verdict-banner'),
    verdictIcon: document.getElementById('verdict-icon'),
    verdictBadge: document.getElementById('verdict-badge'),
    verdictHeading: document.getElementById('verdict-heading'),
    verdictSub: document.getElementById('verdict-sub'),
    probDefectVal: document.getElementById('prob-defect-val'),
    probDefectBar: document.getElementById('prob-defect-bar'),
    probGoodVal: document.getElementById('prob-good-val'),
    probGoodBar: document.getElementById('prob-good-bar'),
    metroLatency: document.getElementById('metro-latency'),
    metroAnomalyType: document.getElementById('metro-anomaly-type'),
    metroDpNoise: document.getElementById('metro-dp-noise'),
    metroActuator: document.getElementById('metro-actuator'),
    thresholdSlider: document.getElementById('threshold-slider'),
    thresholdVal: document.getElementById('threshold-val'),

    // Chart & Stepper
    dpChartCanvas: document.getElementById('dp-chart-canvas'),
    chartTooltip: document.getElementById('chart-tooltip'),
    roundSlider: document.getElementById('round-slider'),
    scrubberRoundNum: document.getElementById('scrubber-round-num'),
    logNumClients: document.getElementById('log-num-clients'),
    logEpsilon: document.getElementById('log-epsilon'),
    logFailures: document.getElementById('log-failures'),
    logWeightTx: document.getElementById('log-weight-tx'),
    roundAnnotationText: document.getElementById('round-annotation-text'),

    // Simulation
    btnSimulateRound: document.getElementById('btn-simulate-round'),
    simStatusBanner: document.getElementById('sim-status-banner'),
    simStatusText: document.getElementById('sim-status-text'),
    nodeCards: [
      document.getElementById('node-factory-a'),
      document.getElementById('node-factory-b'),
      document.getElementById('node-factory-c')
    ],
    packetElements: [
      document.getElementById('packet-a'),
      document.getElementById('packet-b'),
      document.getElementById('packet-c')
    ]
  };

  // =========================================================================
  // 3. INITIALIZATION
  // =========================================================================

  function init() {
    setupNavigation();
    setupFactoryTabs();
    setupSampleTray();
    setupImageUploader();
    setupSensitivitySlider();
    setupCamToggle();
    setupChart();
    setupRoundStepper();
    setupSimulation();

    // Select initial sample
    selectFactory('factory_a');
  }

  // =========================================================================
  // 4. FACTORY & SAMPLE MANAGEMENT
  // =========================================================================

  function selectFactory(factoryId) {
    state.activeFactory = factoryId;
    const factory = FACTORY_DATA[factoryId];
    if (!factory) return;

    // Update active tab buttons
    document.querySelectorAll('.factory-tab').forEach(tab => {
      const isCurrent = tab.dataset.factory === factoryId;
      tab.classList.toggle('active', isCurrent);
      tab.setAttribute('aria-selected', isCurrent);
    });

    // Update HUD Tag
    elements.hudModelTag.textContent = `${factory.name.split(' ')[0]} ${factory.name.split(' ')[1]} (R10)`;

    // Render sample thumbnails
    renderSampleThumbnails(factory);

    // Default to the first sample of the factory
    if (factory.samples.length > 0) {
      selectSample(factory.samples[0]);
    }
  }

  function renderSampleThumbnails(factory) {
    elements.sampleThumbnailTray.innerHTML = '';
    elements.sampleCountLabel.textContent = `${factory.samples.length} items available`;

    factory.samples.forEach(sample => {
      const card = document.createElement('div');
      card.className = `sample-card ${sample.id === (state.activeSample && state.activeSample.id) ? 'active' : ''}`;
      card.id = `sample-${sample.id}`;

      const tagClass = sample.type === 'defect' ? 'tag-defect' : 'tag-good';
      const tagText = sample.type === 'defect' ? 'DEFECT' : 'PASS';

      card.innerHTML = `
        <img src="${sample.image}" alt="${sample.name}" class="sample-thumb-img" loading="lazy">
        <div class="sample-card-info">
          <span class="sample-card-name">${sample.name}</span>
          <span class="sample-card-tag ${tagClass}">${tagText} (${sample.type === 'defect' ? sample.defectProb : sample.goodProb}%)</span>
        </div>
      `;

      card.addEventListener('click', () => {
        selectSample(sample);
      });

      elements.sampleThumbnailTray.appendChild(card);
    });
  }

  function selectSample(sample) {
    state.activeSample = sample;

    // Update active card indicator
    document.querySelectorAll('.sample-card').forEach(card => {
      card.classList.toggle('active', card.id === `sample-${sample.id}`);
    });

    // Trigger visual scan effect
    triggerLaserScan();

    // Load image
    elements.inspectionImage.src = sample.image;
    elements.inspectionImage.onload = () => {
      renderCamHeatmap(sample.cam);
    };

    // Update Telemetry & Decision HUD
    updateTelemetryHUD(sample);
  }

  function triggerLaserScan() {
    state.isScanning = true;
    elements.canvasContainer.classList.add('scanning-active');

    setTimeout(() => {
      state.isScanning = false;
      elements.canvasContainer.classList.remove('scanning-active');
    }, 1400);
  }

  // =========================================================================
  // 5. ANOMALY CAM HEATMAP GENERATOR (CANVAS)
  // =========================================================================

  function renderCamHeatmap(camData) {
    const canvas = elements.heatmapCanvas;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!state.camEnabled || !camData) {
      canvas.style.opacity = '0';
      return;
    }

    canvas.style.opacity = '0.85';

    const { x, y, r, intensity = 0.9 } = camData;

    // Create multi-stop glowing radial gradient to simulate Grad-CAM attention
    const radGrad = ctx.createRadialGradient(x, y, 4, x, y, r);
    radGrad.addColorStop(0.0, `rgba(244, 63, 94, ${intensity * 0.95})`);      // Hot center red
    radGrad.addColorStop(0.3, `rgba(249, 115, 22, ${intensity * 0.85})`);     // Warm orange
    radGrad.addColorStop(0.55, `rgba(234, 179, 8, ${intensity * 0.65})`);     // Anomaly yellow
    radGrad.addColorStop(0.8, `rgba(6, 182, 212, ${intensity * 0.35})`);      // Cool cyan border
    radGrad.addColorStop(1.0, 'rgba(6, 182, 212, 0.0)');                     // Transparent fade

    ctx.fillStyle = radGrad;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();

    // Draw inner high-confidence anomaly core
    const coreGrad = ctx.createRadialGradient(x, y, 0, x, y, r * 0.35);
    coreGrad.addColorStop(0.0, 'rgba(255, 255, 255, 0.85)');
    coreGrad.addColorStop(1.0, 'rgba(244, 63, 94, 0.0)');
    ctx.fillStyle = coreGrad;
    ctx.beginPath();
    ctx.arc(x, y, r * 0.35, 0, Math.PI * 2);
    ctx.fill();
  }

  function setupCamToggle() {
    elements.btnToggleCam.addEventListener('click', () => {
      state.camEnabled = !state.camEnabled;
      elements.btnToggleCam.classList.toggle('active', state.camEnabled);
      if (state.activeSample) {
        renderCamHeatmap(state.activeSample.cam);
      }
    });
  }

  // =========================================================================
  // 6. TELEMETRY & DECISION HUD
  // =========================================================================

  function updateTelemetryHUD(sample) {
    const defectProb = sample.defectProb;
    const goodProb = sample.goodProb;
    const threshold = state.sensitivityThreshold;

    // Probability Bars
    elements.probDefectVal.textContent = `${defectProb.toFixed(2)}%`;
    elements.probDefectBar.style.width = `${defectProb}%`;
    elements.probGoodVal.textContent = `${goodProb.toFixed(2)}%`;
    elements.probGoodBar.style.width = `${goodProb}%`;

    // Metrology readouts
    elements.metroLatency.textContent = sample.latency || '13.4 ms';
    elements.metroAnomalyType.textContent = sample.anomaly || 'None';
    elements.metroActuator.textContent = sample.actuator || 'PASS';

    // Decision logic using sensitivity threshold
    const isDefect = defectProb >= threshold;

    if (isDefect) {
      elements.verdictBanner.className = 'verdict-banner defect';
      elements.verdictBadge.textContent = 'QC ALERT • HIGH RISK';
      elements.verdictHeading.textContent = 'DEFECT DETECTED';
      elements.verdictSub.textContent = `Anomaly confidence (${defectProb.toFixed(1)}%) breaches the ${threshold}% sensitivity limit.`;
      elements.metroActuator.className = 'metro-val action-val';

      elements.verdictIcon.innerHTML = `
        <circle cx="12" cy="12" r="10" />
        <line x1="12" y1="8" x2="12" y2="12" />
        <line x1="12" y1="16" x2="12.01" y2="16" />
      `;
    } else {
      elements.verdictBanner.className = 'verdict-banner good';
      elements.verdictBadge.textContent = 'QC VERIFIED • NOMINAL';
      elements.verdictHeading.textContent = 'QUALITY PASSED';
      elements.verdictSub.textContent = `Workpiece complies with manufacturing tolerances (Good confidence: ${goodProb.toFixed(1)}%).`;
      elements.metroActuator.className = 'metro-val highlight-val';

      elements.verdictIcon.innerHTML = `
        <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
        <polyline points="22 4 12 14.01 9 11.01" />
      `;
    }
  }

  function setupSensitivitySlider() {
    elements.thresholdSlider.addEventListener('input', (e) => {
      const val = parseInt(e.target.value, 10);
      state.sensitivityThreshold = val;
      elements.thresholdVal.textContent = `${val}%`;
      if (state.activeSample) {
        updateTelemetryHUD(state.activeSample);
      }
    });
  }

  // =========================================================================
  // 7. CUSTOM WORKPIECE UPLOADER (DRAG & DROP)
  // =========================================================================

  function setupImageUploader() {
    const dropzone = elements.uploadDropzone;
    const input = elements.fileInput;

    ['dragenter', 'dragover'].forEach(eventName => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        dropzone.classList.add('drag-over');
      });
    });

    ['dragleave', 'drop'].forEach(eventName => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        dropzone.classList.remove('drag-over');
      });
    });

    dropzone.addEventListener('drop', (e) => {
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        processUploadedFile(e.dataTransfer.files[0]);
      }
    });

    input.addEventListener('change', (e) => {
      if (e.target.files && e.target.files[0]) {
        processUploadedFile(e.target.files[0]);
      }
    });
  }

  function processUploadedFile(file) {
    if (!file.type.match('image.*')) {
      alert('Please upload an image file (PNG, JPG, or JPEG).');
      return;
    }

    const reader = new FileReader();
    reader.onload = function (e) {
      const imgDataUrl = e.target.result;

      // Simulated Inference for uploaded sample
      // Produces realistic score variance based on name & size
      const isLikelyDefect = file.size % 2 === 0;
      const defectScore = isLikelyDefect ? 92.5 + (file.size % 70) / 10 : 3.2 + (file.size % 40) / 10;
      const goodScore = 100 - defectScore;

      const customSample = {
        id: `upload_${Date.now()}`,
        name: file.name.substring(0, 16),
        type: isLikelyDefect ? 'defect' : 'good',
        image: imgDataUrl,
        defectProb: defectScore,
        goodProb: goodScore,
        anomaly: isLikelyDefect ? 'Unspecified Surface Deviation' : 'None Detected',
        actuator: isLikelyDefect ? 'REVERT TO INSPECTION QUEUE' : 'PASS TO LOGISTICS',
        latency: `${(11 + (file.size % 50) / 10).toFixed(1)} ms`,
        cam: isLikelyDefect ? { x: 200, y: 190, r: 75, intensity: 0.9 } : null
      };

      selectSample(customSample);
    };
    reader.readAsDataURL(file);
  }

  // =========================================================================
  // 8. INTERACTIVE CANVAS DP BUDGET CHART
  // =========================================================================

  function setupChart() {
    const canvas = elements.dpChartCanvas;
    if (!canvas) return;

    function resizeCanvas() {
      const rect = canvas.parentElement.getBoundingClientRect();
      canvas.width = rect.width * window.devicePixelRatio;
      canvas.height = rect.height * window.devicePixelRatio;
      drawChart();
    }

    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();

    // Chart Hover Tooltip
    canvas.addEventListener('mousemove', (e) => {
      const rect = canvas.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;

      const paddingLeft = 50;
      const paddingRight = 30;
      const chartWidth = rect.width - paddingLeft - paddingRight;

      const roundIndex = Math.round(((mouseX - paddingLeft) / chartWidth) * (TRAINING_ROUNDS_DATA.length - 1));

      if (roundIndex >= 0 && roundIndex < TRAINING_ROUNDS_DATA.length) {
        const item = TRAINING_ROUNDS_DATA[roundIndex];
        const tooltip = elements.chartTooltip;
        tooltip.style.display = 'block';
        tooltip.style.left = `${mouseX}px`;
        tooltip.style.top = `${mouseY}px`;
        tooltip.innerHTML = `
          <strong>Round ${item.round} / 10</strong><br>
          &epsilon; Spent: <strong>${item.epsilon.toFixed(2)}</strong><br>
          Clients: 3 active<br>
          Bound: &delta; = 1e-5
        `;
      }
    });

    canvas.addEventListener('mouseleave', () => {
      elements.chartTooltip.style.display = 'none';
    });
  }

  function drawChart() {
    const canvas = elements.dpChartCanvas;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    const dpr = window.devicePixelRatio || 1;

    ctx.clearRect(0, 0, width, height);

    const padLeft = 60 * dpr;
    const padRight = 30 * dpr;
    const padTop = 30 * dpr;
    const padBottom = 40 * dpr;

    const plotW = width - padLeft - padRight;
    const plotH = height - padTop - padBottom;

    const maxEpsilon = 6.0; // scale y-axis up to 6.0
    const rounds = TRAINING_ROUNDS_DATA.length;

    // Helper functions for coordinates
    const getX = (round) => padLeft + ((round - 1) / (rounds - 1)) * plotW;
    const getY = (val) => padTop + plotH - (val / maxEpsilon) * plotH;

    // 1. Draw Grid lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.06)';
    ctx.lineWidth = 1 * dpr;
    ctx.fillStyle = '#64748b';
    ctx.font = `${11 * dpr}px 'JetBrains Mono', monospace`;
    ctx.textAlign = 'right';

    for (let val = 0; val <= maxEpsilon; val += 1.0) {
      const y = getY(val);
      ctx.beginPath();
      ctx.moveTo(padLeft, y);
      ctx.lineTo(width - padRight, y);
      ctx.stroke();
      ctx.fillText(`${val.toFixed(1)}ε`, padLeft - 10 * dpr, y + 4 * dpr);
    }

    // X-axis labels
    ctx.textAlign = 'center';
    for (let r = 1; r <= 10; r += 1) {
      const x = getX(r);
      ctx.fillText(`R${r}`, x, height - 15 * dpr);
    }

    // 2. Draw Safety Limit Line at epsilon = 5.0
    const limitY = getY(5.0);
    ctx.strokeStyle = 'rgba(244, 63, 94, 0.6)';
    ctx.setLineDash([4 * dpr, 4 * dpr]);
    ctx.lineWidth = 1.5 * dpr;
    ctx.beginPath();
    ctx.moveTo(padLeft, limitY);
    ctx.lineTo(width - padRight, limitY);
    ctx.stroke();
    ctx.setLineDash([]); // Reset line dash

    // 3. Draw Shaded Gradient Area under Cumulative Epsilon curve
    const areaGrad = ctx.createLinearGradient(0, padTop, 0, padTop + plotH);
    areaGrad.addColorStop(0, 'rgba(139, 92, 246, 0.35)');
    areaGrad.addColorStop(1, 'rgba(139, 92, 246, 0.01)');

    ctx.fillStyle = areaGrad;
    ctx.beginPath();
    ctx.moveTo(getX(1), getY(0));
    TRAINING_ROUNDS_DATA.forEach(d => {
      ctx.lineTo(getX(d.round), getY(d.epsilon));
    });
    ctx.lineTo(getX(10), getY(0));
    ctx.closePath();
    ctx.fill();

    // 4. Draw Smooth Line for Epsilon
    ctx.strokeStyle = '#8b5cf6';
    ctx.lineWidth = 3 * dpr;
    ctx.shadowColor = 'rgba(139, 92, 246, 0.6)';
    ctx.shadowBlur = 10 * dpr;

    ctx.beginPath();
    TRAINING_ROUNDS_DATA.forEach((d, i) => {
      const x = getX(d.round);
      const y = getY(d.epsilon);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.shadowBlur = 0; // reset

    // 5. Draw Point Dots
    TRAINING_ROUNDS_DATA.forEach(d => {
      const x = getX(d.round);
      const y = getY(d.epsilon);
      const isSelected = d.round === state.selectedRound;

      ctx.beginPath();
      ctx.arc(x, y, (isSelected ? 6 : 4) * dpr, 0, Math.PI * 2);
      ctx.fillStyle = isSelected ? '#a78bfa' : '#8b5cf6';
      ctx.fill();
      ctx.strokeStyle = '#070a12';
      ctx.lineWidth = 2 * dpr;
      ctx.stroke();

      if (isSelected) {
        ctx.beginPath();
        ctx.arc(x, y, 9 * dpr, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(167, 139, 250, 0.6)';
        ctx.lineWidth = 2 * dpr;
        ctx.stroke();
      }
    });
  }

  // =========================================================================
  // 9. ROUND STEPPER & TIMELINE
  // =========================================================================

  function setupRoundStepper() {
    elements.roundSlider.addEventListener('input', (e) => {
      const round = parseInt(e.target.value, 10);
      selectRound(round);
    });
  }

  function selectRound(roundNum) {
    state.selectedRound = roundNum;
    const data = TRAINING_ROUNDS_DATA[roundNum - 1];
    if (!data) return;

    elements.scrubberRoundNum.textContent = `Round ${roundNum}`;
    elements.logEpsilon.textContent = `${data.epsilon.toFixed(2)}`;
    elements.roundAnnotationText.textContent = data.annotation;

    // Redraw chart to update selected highlight
    drawChart();
  }

  // =========================================================================
  // 10. FEDERATED AGGREGATION MESH SIMULATION
  // =========================================================================

  function setupSimulation() {
    elements.btnSimulateRound.addEventListener('click', runFederatedSimulation);
  }

  function runFederatedSimulation() {
    if (state.isSimulatingRound) return;
    state.isSimulatingRound = true;

    const btn = elements.btnSimulateRound;
    btn.disabled = true;
    btn.style.opacity = '0.6';

    const statusBanner = elements.simStatusText;

    // Step 1: Local edge computation
    statusBanner.textContent = 'Step 1: Factories computing local YOLOv8 gradients on private CCTV feeds...';
    elements.nodeCards.forEach(card => card.classList.add('simulating'));

    setTimeout(() => {
      // Step 2: Gradient clipping & Differential Privacy noise injection
      statusBanner.textContent = 'Step 2: Injecting calibrated Gaussian noise (σ=0.8) into gradient tensors...';
      elements.packetElements.forEach(packet => packet.classList.add('packet-animating'));
    }, 1200);

    setTimeout(() => {
      // Step 3: Flower FedAvg Aggregator weight fusion
      statusBanner.textContent = 'Step 3: Flower FedAvg server aggregating noise-injected ΔW weights...';
    }, 2400);

    setTimeout(() => {
      // Step 4: Broadcast global model update
      statusBanner.textContent = 'Step 4: Global YOLOv8 model weights synchronized back to edge nodes!';
      elements.nodeCards.forEach(card => card.classList.remove('simulating'));
      elements.packetElements.forEach(packet => packet.classList.remove('packet-animating'));
    }, 3600);

    setTimeout(() => {
      statusBanner.textContent = 'Network Idle • Global Model Weights Synchronized (Round 10)';
      btn.disabled = false;
      btn.style.opacity = '1';
      state.isSimulatingRound = false;
    }, 4500);
  }

  // =========================================================================
  // 11. NAVIGATION & FACTORY TABS
  // =========================================================================

  function setupNavigation() {
    const navLinks = document.querySelectorAll('.nav-link');
    window.addEventListener('scroll', () => {
      const scrollPos = window.scrollY + 120;
      const sections = ['inspection-section', 'topology-section', 'metrics-section', 'architecture-section', 'research-section'];

      sections.forEach(secId => {
        const sec = document.getElementById(secId);
        if (!sec) return;
        const top = sec.offsetTop;
        const height = sec.offsetHeight;

        if (scrollPos >= top && scrollPos < top + height) {
          navLinks.forEach(link => {
            link.classList.toggle('active', link.getAttribute('href') === `#${secId}`);
          });
        }
      });
    });
  }

  function setupFactoryTabs() {
    document.querySelectorAll('.factory-tab').forEach(tab => {
      tab.addEventListener('click', () => {
        const factoryId = tab.dataset.factory;
        selectFactory(factoryId);
      });
    });
  }

  // Run on DOM Content Loaded
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
