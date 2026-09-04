/**
 * QURA - Hybrid Quantum-Classical UI Client Logic (Black, Red & White Theme v3.0)
 */

let featureNames = [];
let presetSamples = {};
let currentFeatures = [];
let activeSampleKey = "sample_0_malignant";
let lastPredictionResult = null;

let radarChartInstance = null;
let qubitBarChartInstance = null;
let importanceBarChartInstance = null;

// Bloch Sphere State
let blochTheta = Math.PI / 3;
let blochPhi = Math.PI / 4;
let targetBlochTheta = Math.PI / 3;
let targetBlochPhi = Math.PI / 4;

document.addEventListener("DOMContentLoaded", async () => {
  initBackgroundAnimation();
  initCharts();
  initBlochSphere();
  renderCircuitTopology();
  setupTabs();
  setupEvents();
  setupBatchUpload();

  await loadMetadata();
  await loadSamples();
  await loadFeatureImportance();

  // Run initial prediction on default sample
  if (presetSamples[activeSampleKey]) {
    loadPresetIntoInputs(presetSamples[activeSampleKey]);
    runDiagnosis(true);
  }
});

// -------------------- Interactive Background Animation (Red & White on Black) --------------------
function initBackgroundAnimation() {
  const canvas = document.getElementById("quantum-bg-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");

  let width = canvas.width = window.innerWidth;
  let height = canvas.height = window.innerHeight;

  window.addEventListener("resize", () => {
    width = canvas.width = window.innerWidth;
    height = canvas.height = window.innerHeight;
  });

  const particles = [];
  const numParticles = Math.floor(Math.min(width, 1400) / 26);

  for (let i = 0; i < numParticles; i++) {
    const isRed = Math.random() > 0.45;
    particles.push({
      x: Math.random() * width,
      y: Math.random() * height,
      vx: (Math.random() - 0.5) * 0.35,
      vy: (Math.random() - 0.5) * 0.35,
      radius: Math.random() * 2 + 1,
      color: isRed ? "rgba(255, 42, 75, " : "rgba(255, 255, 255, ",
      alpha: Math.random() * 0.4 + 0.3
    });
  }

  function render() {
    ctx.clearRect(0, 0, width, height);

    for (let i = 0; i < particles.length; i++) {
      const p = particles[i];
      p.x += p.vx;
      p.y += p.vy;

      if (p.x < 0) p.x = width;
      if (p.x > width) p.x = 0;
      if (p.y < 0) p.y = height;
      if (p.y > height) p.y = 0;

      ctx.fillStyle = `${p.color}${p.alpha})`;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
      ctx.fill();

      for (let j = i + 1; j < particles.length; j++) {
        const p2 = particles[j];
        const dx = p.x - p2.x;
        const dy = p.y - p2.y;
        const dist = Math.sqrt(dx * dx + dy * dy);

        if (dist < 130) {
          const lineAlpha = (1 - dist / 130) * 0.12;
          ctx.strokeStyle = `rgba(255, 42, 75, ${lineAlpha})`;
          ctx.beginPath();
          ctx.moveTo(p.x, p.y);
          ctx.lineTo(p2.x, p2.y);
          ctx.stroke();
        }
      }
    }
    requestAnimationFrame(render);
  }
  render();
}

// -------------------- 3D Interactive Bloch Sphere (Red & White) --------------------
function initBlochSphere() {
  const canvas = document.getElementById("blochSphereCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");

  let rotAngle = 0;

  function drawSphere() {
    const w = canvas.width = canvas.parentElement.clientWidth;
    const h = canvas.height = canvas.parentElement.clientHeight;
    const cx = w / 2;
    const cy = h / 2;
    const r = Math.min(w, h) * 0.36;

    ctx.clearRect(0, 0, w, h);
    rotAngle += 0.006;

    // Ease current angle towards target
    blochTheta += (targetBlochTheta - blochTheta) * 0.08;
    blochPhi += (targetBlochPhi - blochPhi) * 0.08;

    // Main Sphere Outer Circle (White)
    ctx.strokeStyle = "rgba(255, 255, 255, 0.2)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.stroke();

    // Equator Ellipse (Red)
    ctx.strokeStyle = "rgba(255, 42, 75, 0.35)";
    ctx.beginPath();
    ctx.ellipse(cx, cy, r, r * 0.35, 0, 0, Math.PI * 2);
    ctx.stroke();

    // Meridian Ellipse (White)
    ctx.strokeStyle = "rgba(255, 255, 255, 0.15)";
    ctx.beginPath();
    ctx.ellipse(cx, cy, r * 0.35, r, 0, 0, Math.PI * 2);
    ctx.stroke();

    // Z-Axis
    ctx.strokeStyle = "rgba(255, 255, 255, 0.35)";
    ctx.beginPath();
    ctx.moveTo(cx, cy - r - 15);
    ctx.lineTo(cx, cy + r + 15);
    ctx.stroke();

    // Labels |0> (White) and |1> (Red)
    ctx.fillStyle = "#ffffff";
    ctx.font = "bold 12px 'Space Grotesk'";
    ctx.fillText("|0⟩", cx - 8, cy - r - 20);
    ctx.fillStyle = "#ff2a4b";
    ctx.fillText("|1⟩", cx - 8, cy + r + 28);

    // Compute Vector tip on Sphere
    const currentTheta = blochTheta;
    const currentPhi = blochPhi + rotAngle;

    const sx = Math.sin(currentTheta) * Math.cos(currentPhi);
    const sy = Math.cos(currentTheta);
    const sz = Math.sin(currentTheta) * Math.sin(currentPhi);

    const px = cx + sx * r;
    const py = cy - sy * r + sz * (r * 0.2);

    // Draw State Vector Line (Crimson Red Glow)
    ctx.strokeStyle = "#ff2a4b";
    ctx.lineWidth = 3;
    ctx.shadowColor = "#ff2a4b";
    ctx.shadowBlur = 16;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(px, py);
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Draw Vector Tip (Pure White)
    ctx.fillStyle = "#ffffff";
    ctx.beginPath();
    ctx.arc(px, py, 5, 0, Math.PI * 2);
    ctx.fill();

    // State Label
    ctx.fillStyle = "#ffffff";
    ctx.font = "bold 11px 'JetBrains Mono'";
    ctx.fillText("|ψ⟩", px + 8, py - 4);

    requestAnimationFrame(drawSphere);
  }
  drawSphere();
}

// -------------------- Chart Initialization --------------------
function initCharts() {
  // 1. Radar Chart (Crimson Red & White)
  const ctxRadar = document.getElementById("quantumRadarChart").getContext("2d");
  radarChartInstance = new Chart(ctxRadar, {
    type: "radar",
    data: {
      labels: ["Q0 (PCA 1)", "Q1 (PCA 2)", "Q2 (PCA 3)", "Q3 (PCA 4)", "Q4 (PCA 5)", "Q5 (PCA 6)", "Q6 (PCA 7)", "Q7 (PCA 8)"],
      datasets: [{
        label: "Quantum Rotation Angle (rad)",
        data: [0, 0, 0, 0, 0, 0, 0, 0],
        backgroundColor: "rgba(255, 42, 75, 0.2)",
        borderColor: "#ff2a4b",
        pointBackgroundColor: "#ffffff",
        pointBorderColor: "#ff2a4b",
        pointHoverBackgroundColor: "#ff2a4b",
        pointHoverBorderColor: "#ffffff",
        borderWidth: 2,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        r: {
          angleLines: { color: "rgba(255, 255, 255, 0.09)" },
          grid: { color: "rgba(255, 255, 255, 0.07)" },
          pointLabels: { color: "#a1a1aa", font: { family: "Plus Jakarta Sans", size: 11, weight: "600" } },
          ticks: { backdropColor: "transparent", color: "#71717a", min: 0, max: Math.PI }
        }
      },
      plugins: {
        legend: { display: false }
      }
    }
  });

  // 2. Qubit Expectations Bar Chart (Red & White)
  const ctxBar = document.getElementById("qubitBarChart").getContext("2d");
  qubitBarChartInstance = new Chart(ctxBar, {
    type: "bar",
    data: {
      labels: ["Q0", "Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7"],
      datasets: [{
        label: "Pauli-Z Expectation",
        data: [0, 0, 0, 0, 0, 0, 0, 0],
        backgroundColor: "rgba(255, 42, 75, 0.55)",
        borderColor: "#ff2a4b",
        borderWidth: 1.5,
        borderRadius: 8,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          min: -1.0,
          max: 1.0,
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          ticks: { color: "#a1a1aa", font: { family: "JetBrains Mono", size: 11 } }
        },
        x: {
          grid: { display: false },
          ticks: { color: "#a1a1aa", font: { family: "JetBrains Mono", size: 12, weight: "600" } }
        }
      },
      plugins: {
        legend: { display: false }
      }
    }
  });

  // 3. Feature Importance Horizontal Bar Chart (Red & White)
  const ctxImp = document.getElementById("importanceBarChart").getContext("2d");
  importanceBarChartInstance = new Chart(ctxImp, {
    type: "bar",
    data: {
      labels: [],
      datasets: [{
        label: "PCA Quantum Loading Magnitude",
        data: [],
        backgroundColor: "rgba(255, 42, 75, 0.5)",
        borderColor: "#ff2a4b",
        borderWidth: 1.5,
        borderRadius: 6,
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          ticks: { color: "#a1a1aa", font: { family: "JetBrains Mono", size: 11 } }
        },
        y: {
          grid: { display: false },
          ticks: { color: "#ffffff", font: { family: "Plus Jakarta Sans", size: 12, weight: "600" } }
        }
      },
      plugins: {
        legend: { display: false }
      }
    }
  });
}

// -------------------- Circuit Topology Renderer --------------------
function renderCircuitTopology() {
  const container = document.getElementById("circuit-wires-list");
  if (!container) return;
  container.innerHTML = "";

  for (let q = 0; q < 8; q++) {
    const row = document.createElement("div");
    row.className = "wire-row";
    row.id = `wire-row-${q}`;

    row.innerHTML = `
      <div class="wire-label">|0⟩ q[${q}]</div>
      <div class="gate-block">RY(θ${q})</div>
      <div class="gate-block entangle">Rot + CNOT</div>
      <div class="gate-block">RY(θ${q})</div>
      <div class="gate-block entangle">Rot + CNOT</div>
      <div class="gate-block">RY(θ${q})</div>
      <div class="gate-block entangle">Rot + CNOT</div>
      <div class="gate-block measure">⟨Z${q}⟩</div>
    `;
    container.appendChild(row);
  }
}

function triggerCircuitPulse() {
  for (let q = 0; q < 8; q++) {
    const row = document.getElementById(`wire-row-${q}`);
    if (row) {
      setTimeout(() => {
        row.classList.remove("pulsing");
        void row.offsetWidth;
        row.classList.add("pulsing");
      }, q * 40);
    }
  }
}

// -------------------- Tab Navigation --------------------
function setupTabs() {
  const tabs = document.querySelectorAll(".tab-btn");
  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      tabs.forEach(t => t.classList.remove("active"));
      tab.classList.add("active");

      const targetTab = tab.getAttribute("data-tab");
      document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));
      document.getElementById(targetTab).classList.add("active");
    });
  });
}

// -------------------- Events & Handlers --------------------
function setupEvents() {
  document.getElementById("btn-run-all").addEventListener("click", () => {
    collectInputsFromUI();
    runDiagnosis();
  });

  document.getElementById("btn-reset-defaults").addEventListener("click", () => {
    if (presetSamples[activeSampleKey]) {
      loadPresetIntoInputs(presetSamples[activeSampleKey]);
      runDiagnosis();
      showToast("Reset features to selected preset.", "info");
    }
  });

  document.getElementById("btn-random-sample").addEventListener("click", () => {
    const keys = Object.keys(presetSamples);
    const randomKey = keys[Math.floor(Math.random() * keys.length)];
    setActiveChip(randomKey);
  });

  // Modal handlers
  const modal = document.getElementById("report-modal");
  document.getElementById("btn-export-report").addEventListener("click", () => {
    populateModalReport();
    modal.classList.add("open");
  });

  document.getElementById("btn-close-modal").addEventListener("click", () => {
    modal.classList.remove("open");
  });

  document.getElementById("btn-print-report").addEventListener("click", () => {
    window.print();
  });

  modal.addEventListener("click", (e) => {
    if (e.target === modal) modal.classList.remove("open");
  });
}

// -------------------- Toast Notifications --------------------
function showToast(message, type = "success") {
  const container = document.getElementById("toast-container");
  if (!container) return;

  const toast = document.createElement("div");
  toast.className = "toast";

  const icon = type === "success" ? "fa-circle-check" : (type === "warning" ? "fa-triangle-exclamation" : "fa-circle-info");
  const color = type === "success" ? "#ff2a4b" : (type === "warning" ? "#f59e0b" : "#ffffff");

  toast.innerHTML = `<i class="fa-solid ${icon}" style="color: ${color}; font-size: 16px;"></i> <span>${message}</span>`;
  container.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = "0";
    toast.style.transform = "translateX(50px)";
    toast.style.transition = "all 0.3s ease";
    setTimeout(() => toast.remove(), 300);
  }, 3200);
}

// -------------------- Batch CSV Upload Handler --------------------
function setupBatchUpload() {
  const dropzone = document.getElementById("csv-dropzone");
  const fileInput = document.getElementById("csv-file-input");

  dropzone.addEventListener("click", () => fileInput.click());

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files.length) {
      handleCsvFile(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length) {
      handleCsvFile(e.target.files[0]);
    }
  });
}

function handleCsvFile(file) {
  showToast(`Uploading and parsing ${file.name}...`, "info");
  const reader = new FileReader();
  reader.onload = async (e) => {
    const text = e.target.result;
    const rows = parseCsvToRows(text);

    if (rows.length === 0) {
      showToast("No valid rows found in CSV. Expected 30 numerical features.", "warning");
      return;
    }

    try {
      const res = await fetch("/api/batch_predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ rows: rows })
      });
      const batchResult = await res.json();
      renderBatchResults(batchResult);
      showToast(`Analyzed ${batchResult.total_cases} patient records successfully!`, "success");
    } catch (err) {
      console.error("Batch processing error:", err);
      showToast("Failed to process batch CSV.", "warning");
    }
  };
  reader.readAsText(file);
}

function parseCsvToRows(csvText) {
  const lines = csvText.trim().split("\n");
  const rows = [];

  for (let i = 1; i < lines.length; i++) {
    const parts = lines[i].split(",").map(p => p.trim());
    if (parts.length >= 30) {
      const id = isNaN(parts[0]) ? parts[0] : `Patient #${i}`;
      const featureValues = parts.slice(-30).map(v => parseFloat(v) || 0.0);
      if (featureValues.length === 30) {
        rows.push({ id: id, features: featureValues });
      }
    }
  }
  return rows;
}

function renderBatchResults(data) {
  const panel = document.getElementById("batch-summary-panel");
  panel.style.display = "flex";

  document.getElementById("b-stat-total").innerText = data.total_cases;
  document.getElementById("b-stat-malig").innerText = data.malignant_count;
  document.getElementById("b-stat-benign").innerText = data.benign_count;
  document.getElementById("b-stat-consensus").innerText = `${data.consensus_rate}%`;

  const tbody = document.getElementById("batch-table-body");
  tbody.innerHTML = "";

  data.results.forEach(item => {
    const tr = document.createElement("tr");
    const qClass = item.quantum.label.toLowerCase();

    tr.innerHTML = `
      <td><strong>${item.patient_id}</strong></td>
      <td><span class="chip-status-dot" style="display:inline-block; margin-right:6px; background:${qClass === 'malignant' ? '#ff2a4b' : '#ffffff'};"></span>${item.quantum.label}</td>
      <td class="mono">${item.quantum.confidence}%</td>
      <td>${item.classical.label}</td>
      <td class="mono">${item.classical.confidence}%</td>
      <td>
        <span class="consensus-tag" style="padding: 2px 8px; font-size: 11px; ${item.consensus ? '' : 'color:#f59e0b; border-color:rgba(245,158,11,0.4);'}">
          ${item.consensus ? 'Agreed' : 'Divergent'}
        </span>
      </td>
    `;
    tbody.appendChild(tr);
  });
}

// -------------------- API Requests --------------------
async function loadMetadata() {
  try {
    const res = await fetch("/api/metadata");
    const data = await res.json();
    featureNames = data.feature_names || [];
    renderFeatureInputs();
  } catch (err) {
    console.error("Failed to load metadata:", err);
  }
}

async function loadSamples() {
  try {
    const res = await fetch("/api/samples");
    presetSamples = await res.json();
    bindPresetChips();
  } catch (err) {
    console.error("Failed to load samples:", err);
  }
}

async function loadFeatureImportance() {
  try {
    const res = await fetch("/api/feature_importance");
    const data = await res.json();
    const top10 = data.rankings.slice(0, 10);

    if (importanceBarChartInstance) {
      importanceBarChartInstance.data.labels = top10.map(item => item.feature);
      importanceBarChartInstance.data.datasets[0].data = top10.map(item => item.importance);
      importanceBarChartInstance.update();
    }
  } catch (err) {
    console.error("Failed to load feature importance:", err);
  }
}

function bindPresetChips() {
  const chips = document.querySelectorAll(".chip[data-sample]");
  chips.forEach(chip => {
    chip.addEventListener("click", () => {
      const sampleKey = chip.getAttribute("data-sample");
      setActiveChip(sampleKey);
    });
  });
}

function setActiveChip(sampleKey) {
  activeSampleKey = sampleKey;
  document.querySelectorAll(".chip").forEach(c => c.classList.remove("active"));
  const activeEl = document.querySelector(`.chip[data-sample="${sampleKey}"]`);
  if (activeEl) activeEl.classList.add("active");

  if (presetSamples[sampleKey]) {
    loadPresetIntoInputs(presetSamples[sampleKey]);
    runDiagnosis();
    showToast(`Loaded ${presetSamples[sampleKey].name}`, "info");
  }
}

// -------------------- Dynamic UI Feature Inputs with Sliders --------------------
function renderFeatureInputs() {
  const gridMean = document.getElementById("grid-mean");
  const gridSe = document.getElementById("grid-se");
  const gridWorst = document.getElementById("grid-worst");

  gridMean.innerHTML = "";
  gridSe.innerHTML = "";
  gridWorst.innerHTML = "";

  featureNames.forEach((name, idx) => {
    const card = document.createElement("div");
    card.className = "input-card";

    const topRow = document.createElement("div");
    topRow.className = "input-top-row";

    const label = document.createElement("label");
    label.title = name;
    label.innerText = name.replace(/_/g, " ");

    const numInput = document.createElement("input");
    numInput.type = "number";
    numInput.step = "any";
    numInput.id = `feat-${idx}`;
    numInput.className = "num-input";
    numInput.value = "0.0";

    const slider = document.createElement("input");
    slider.type = "range";
    slider.min = "0";
    slider.max = "100";
    slider.step = "0.1";
    slider.id = `slider-${idx}`;
    slider.className = "range-slider";

    // Synchronization
    numInput.addEventListener("input", () => {
      slider.value = numInput.value;
      document.querySelectorAll(".chip").forEach(c => c.classList.remove("active"));
    });

    slider.addEventListener("input", () => {
      numInput.value = slider.value;
      document.querySelectorAll(".chip").forEach(c => c.classList.remove("active"));
    });

    topRow.appendChild(label);
    topRow.appendChild(numInput);
    card.appendChild(topRow);
    card.appendChild(slider);

    if (idx < 10) {
      gridMean.appendChild(card);
    } else if (idx < 20) {
      gridSe.appendChild(card);
    } else {
      gridWorst.appendChild(card);
    }
  });
}

function loadPresetIntoInputs(sampleObj) {
  currentFeatures = sampleObj.features;
  currentFeatures.forEach((val, idx) => {
    const numInput = document.getElementById(`feat-${idx}`);
    const slider = document.getElementById(`slider-${idx}`);
    if (numInput) numInput.value = val.toFixed(4);
    if (slider) {
      slider.max = Math.max(100, Math.ceil(val * 1.8)).toString();
      slider.value = val.toFixed(4);
    }
  });
}

function collectInputsFromUI() {
  currentFeatures = [];
  for (let i = 0; i < featureNames.length; i++) {
    const el = document.getElementById(`feat-${i}`);
    currentFeatures.push(el ? parseFloat(el.value) || 0.0 : 0.0);
  }
}

// -------------------- Animated Number Counter Helper --------------------
function animateNumber(elementId, startVal, endVal, duration = 600, suffix = "") {
  const el = document.getElementById(elementId);
  if (!el) return;

  const startTime = performance.now();
  function update(currentTime) {
    const elapsed = currentTime - startTime;
    const progress = Math.min(elapsed / duration, 1);
    const easeProgress = 1 - Math.pow(1 - progress, 3);
    const current = startVal + (endVal - startVal) * easeProgress;

    el.innerText = `${current.toFixed(1)}${suffix}`;
    if (progress < 1) {
      requestAnimationFrame(update);
    }
  }
  requestAnimationFrame(update);
}

// -------------------- Prediction Execution --------------------
async function runDiagnosis(silent = false) {
  if (currentFeatures.length !== 30) {
    collectInputsFromUI();
  }

  const btn = document.getElementById("btn-run-all");
  btn.disabled = true;
  btn.innerHTML = `<span class="btn-icon"><i class="fa-solid fa-spinner fa-spin"></i></span><span>Simulating Circuits...</span>`;

  triggerCircuitPulse();

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ features: currentFeatures })
    });

    const result = await res.json();
    lastPredictionResult = result;
    updateUIWithResults(result);

    if (!silent) {
      showToast(`Dual diagnosis complete (${result.quantum.label} • ${result.quantum.confidence}%)`, "success");
    }
  } catch (err) {
    console.error("Diagnosis error:", err);
    showToast("Diagnosis simulation error.", "warning");
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<span class="btn-icon"><i class="fa-solid fa-play"></i></span><span>Run Dual Diagnosis</span>`;
  }
}

function updateUIWithResults(res) {
  const { quantum, classical, consensus } = res;

  // Quantum UI Elements
  const qVerdict = document.getElementById("quantum-verdict");
  qVerdict.innerText = quantum.label;
  qVerdict.className = `verdict-label ${quantum.label.toLowerCase()}`;

  animateNumber("quantum-confidence-val", 50.0, quantum.confidence, 600, "%");
  document.getElementById("quantum-progress-bar").style.width = `${quantum.confidence}%`;
  document.getElementById("quantum-raw-score").innerText = quantum.raw_score.toFixed(3);
  document.getElementById("quantum-latency").innerText = `${quantum.latency_ms} ms`;

  // Classical UI Elements
  const cVerdict = document.getElementById("classical-verdict");
  cVerdict.innerText = classical.label;
  cVerdict.className = `verdict-label ${classical.label.toLowerCase()}`;

  animateNumber("classical-confidence-val", 50.0, classical.confidence, 600, "%");
  document.getElementById("classical-progress-bar").style.width = `${classical.confidence}%`;
  document.getElementById("classical-prob-malig").innerText = `${classical.probability_malignant}%`;
  document.getElementById("classical-prob-benign").innerText = `${classical.probability_benign}%`;

  // Consensus Status
  const iconEl = document.getElementById("consensus-icon");
  const descEl = document.getElementById("consensus-desc");
  const statusEl = document.getElementById("consensus-status");

  if (consensus) {
    iconEl.innerHTML = `<i class="fa-solid fa-check-double"></i>`;
    iconEl.style.background = "rgba(255, 42, 75, 0.18)";
    iconEl.style.color = "#ff2a4b";
    descEl.innerText = `Full Consensus: Both models agree on ${quantum.label} pathology.`;
    statusEl.innerText = "100% Agreement";
    statusEl.style.borderColor = "rgba(255, 42, 75, 0.45)";
    statusEl.style.color = "#ff2a4b";
  } else {
    iconEl.innerHTML = `<i class="fa-solid fa-triangle-exclamation"></i>`;
    iconEl.style.background = "rgba(245, 158, 11, 0.18)";
    iconEl.style.color = "#f59e0b";
    descEl.innerText = `Divergence: Quantum predicts ${quantum.label}, Classical predicts ${classical.label}.`;
    statusEl.innerText = "Model Divergence";
    statusEl.style.borderColor = "rgba(245, 158, 11, 0.45)";
    statusEl.style.color = "#f59e0b";
  }

  // Update Bloch Sphere Target Orientation from Qubit 0 rotation
  if (quantum.quantum_angles_rad && quantum.quantum_angles_rad.length > 0) {
    targetBlochTheta = quantum.quantum_angles_rad[0];
    targetBlochPhi = quantum.quantum_angles_rad[1] || 0.5;
    document.getElementById("bloch-theta").innerText = `θ: ${targetBlochTheta.toFixed(2)} rad`;
    document.getElementById("bloch-phi").innerText = `φ: ${targetBlochPhi.toFixed(2)} rad`;
  }

  // Update Charts
  if (radarChartInstance && quantum.quantum_angles_rad) {
    radarChartInstance.data.datasets[0].data = quantum.quantum_angles_rad;
    radarChartInstance.update();
  }

  if (qubitBarChartInstance && quantum.qubit_expectations) {
    qubitBarChartInstance.data.datasets[0].data = quantum.qubit_expectations;
    qubitBarChartInstance.data.datasets[0].backgroundColor = quantum.qubit_expectations.map(val => 
      val >= 0 ? "rgba(255, 255, 255, 0.75)" : "rgba(255, 42, 75, 0.7)"
    );
    qubitBarChartInstance.data.datasets[0].borderColor = quantum.qubit_expectations.map(val => 
      val >= 0 ? "#ffffff" : "#ff2a4b"
    );
    qubitBarChartInstance.update();
  }
}

// -------------------- Modal Report Generator --------------------
function populateModalReport() {
  if (!lastPredictionResult) return;
  const { quantum, classical } = lastPredictionResult;

  const currentSampleName = presetSamples[activeSampleKey]?.name || "Custom Biopsy Evaluation";
  document.getElementById("rep-patient-id").innerText = currentSampleName;
  document.getElementById("rep-date").innerText = new Date().toLocaleString();

  const vLabel = document.getElementById("rep-primary-verdict");
  vLabel.innerText = quantum.label.toUpperCase();
  vLabel.style.color = quantum.label === "Malignant" ? "#ff2a4b" : "#ffffff";

  document.getElementById("rep-q-conf").innerText = `${quantum.confidence}%`;
  document.getElementById("rep-c-conf").innerText = `${classical.confidence}%`;
}
