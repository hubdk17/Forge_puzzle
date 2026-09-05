// DataForge Research Workbench Client Application

let currentPuzzle = null;
let currentDemosCount = 5;
let currentKSteps = 5;

// Initialization
document.addEventListener('DOMContentLoaded', () => {
  setupTabs();
  setupSliders();
  setupEventListeners();
  loadTechNote();
  fetchPuzzlesAndLoadFirst();
});

// Tab Switching
function setupTabs() {
  const tabs = document.querySelectorAll('.nav-tab');
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      tabs.forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));

      tab.classList.add('active');
      const target = document.getElementById(`pane-${tab.dataset.tab}`);
      if (target) target.classList.add('active');

      if (tab.dataset.tab === 'state-trace' && currentPuzzle) {
        runStateTrace();
      }
    });
  });
}

// Sliders
function setupSliders() {
  const demosSlider = document.getElementById('demos-slider');
  const demosVal = document.getElementById('demos-count-val');
  demosSlider.addEventListener('input', (e) => {
    currentDemosCount = parseInt(e.target.value);
    demosVal.textContent = currentDemosCount;
    if (currentPuzzle) renderDemos(currentPuzzle.demos.slice(0, currentDemosCount));
  });

  const kSlider = document.getElementById('k-steps-slider');
  const kVal = document.getElementById('k-steps-val');
  kSlider.addEventListener('input', (e) => {
    currentKSteps = parseInt(e.target.value);
    kVal.textContent = currentKSteps;
    document.getElementById('opt-steps-label').textContent = `${currentKSteps} SGD Steps`;
  });
}

function setupEventListeners() {
  document.getElementById('btn-next-puzzle').addEventListener('click', fetchNextPuzzle);
  document.getElementById('btn-run-arena').addEventListener('click', runComparison);
  document.getElementById('split-select').addEventListener('change', fetchPuzzlesAndLoadFirst);
  document.getElementById('rule-filter').addEventListener('change', fetchPuzzlesAndLoadFirst);
  document.getElementById('btn-refresh-trace').addEventListener('click', runStateTrace);
}

// Grid Renderer
function renderGrid(matrix, containerId, sizeClass = 'large') {
  const container = document.getElementById(containerId);
  if (!container) return;
  container.innerHTML = '';

  const gridEl = document.createElement('div');
  gridEl.className = `grid-5x5 ${sizeClass}`;

  for (let r = 0; r < 5; r++) {
    for (let c = 0; c < 5; c++) {
      const cell = document.createElement('div');
      const val = matrix && matrix[r] ? matrix[r][c] : 0;
      cell.className = `cell color-${val}`;
      cell.title = `(${r}, ${c}): color ${val}`;
      gridEl.appendChild(cell);
    }
  }

  container.appendChild(gridEl);
}

// Render Demonstrations Row
function renderDemos(demos) {
  const container = document.getElementById('demos-container');
  container.innerHTML = '';

  demos.forEach((d, idx) => {
    const item = document.createElement('div');
    item.className = 'demo-pair-item';

    const label = document.createElement('div');
    label.className = 'demo-num';
    label.textContent = `Demo ${idx + 1}`;

    const gridsWrap = document.createElement('div');
    gridsWrap.className = 'demo-grids';

    const inWrap = document.createElement('div');
    inWrap.id = `demo-in-${idx}`;
    const arrow = document.createElement('span');
    arrow.className = 'demo-arrow';
    arrow.textContent = '→';
    const outWrap = document.createElement('div');
    outWrap.id = `demo-out-${idx}`;

    gridsWrap.appendChild(inWrap);
    gridsWrap.appendChild(arrow);
    gridsWrap.appendChild(outWrap);

    item.appendChild(label);
    item.appendChild(gridsWrap);
    container.appendChild(item);

    renderGrid(d.input, `demo-in-${idx}`, 'small');
    renderGrid(d.output, `demo-out-${idx}`, 'small');
  });
}

// Fetch Puzzles
async function fetchPuzzlesAndLoadFirst() {
  const split = document.getElementById('split-select').value;
  const rule = document.getElementById('rule-filter').value;
  let url = `/api/puzzles?split=${split}&limit=20`;
  if (rule !== 'all') url += `&rule=${rule}`;

  try {
    const res = await fetch(url);
    const data = await res.json();
    if (data.puzzles && data.puzzles.length > 0) {
      window.cachedPuzzles = data.puzzles;
      window.puzzleIndex = 0;
      loadPuzzle(data.puzzles[0]);
    }
  } catch (err) {
    console.error('Failed to load puzzles:', err);
  }
}

function fetchNextPuzzle() {
  if (!window.cachedPuzzles || window.cachedPuzzles.length === 0) return;
  window.puzzleIndex = (window.puzzleIndex + 1) % window.cachedPuzzles.length;
  loadPuzzle(window.cachedPuzzles[window.puzzleIndex]);
}

function loadPuzzle(puzzle) {
  currentPuzzle = puzzle;

  // Metadata
  document.getElementById('puzzle-id').textContent = puzzle.id || 'custom_puzzle';
  document.getElementById('rule-badge').textContent = puzzle.rule_type;
  
  const paramStr = puzzle.rule_params 
    ? Object.entries(puzzle.rule_params).map(([k, v]) => `${k}: ${v}`).join(', ')
    : 'None';
  document.getElementById('rule-param').textContent = paramStr;

  // Render Demos
  renderDemos(puzzle.demos.slice(0, currentDemosCount));

  // Render Test Inputs and Ground Truth
  renderGrid(puzzle.test_pair.input, 'test-input-grid', 'large');
  renderGrid(puzzle.test_pair.output, 'test-target-grid', 'large');

  // Reset Prediction displays
  document.getElementById('ctx-pred-grid').innerHTML = '<div class="grid-placeholder">Ready to Run</div>';
  document.getElementById('opt-pred-grid').innerHTML = '<div class="grid-placeholder">Ready to Run</div>';
  document.getElementById('ctx-match-badge').textContent = 'Pending';
  document.getElementById('ctx-match-badge').className = 'stat-badge';
  document.getElementById('opt-match-badge').textContent = 'Pending';
  document.getElementById('opt-match-badge').className = 'stat-badge';

  // Automatically execute comparison for snappy feel
  runComparison();
}

// Run Model Arena Comparison
async function runComparison() {
  if (!currentPuzzle) return;

  const btn = document.getElementById('btn-run-arena');
  btn.disabled = true;
  btn.innerHTML = '<span>⚡ Running Inference...</span>';

  try {
    const res = await fetch('/api/compare', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        puzzle: currentPuzzle,
        num_demos: currentDemosCount,
        K: currentKSteps,
        inner_lr: 0.05
      })
    });
    const data = await res.json();

    // 1. Context Model Result
    if (data.context) {
      renderGrid(data.context.prediction, 'ctx-pred-grid', 'large');
      const isMatch = data.context.correct;
      const matchBadge = document.getElementById('ctx-match-badge');
      matchBadge.textContent = isMatch ? '✓ EXACT MATCH (100%)' : '✗ CELL-LEVEL GUESS';
      matchBadge.className = `stat-badge ${isMatch ? 'match' : 'mismatch'}`;

      document.getElementById('ctx-latency').textContent = `${data.context.latency_ms.toFixed(2)} ms`;
      document.getElementById('ctx-state-norm').textContent = data.context.final_state_norm.toFixed(4);

      // Table update
      document.getElementById('table-ctx-lat').textContent = `${data.context.latency_ms.toFixed(2)} ms`;
      document.getElementById('table-ctx-exact').textContent = isMatch ? '100% (Exact Match)' : 'Partial Cell Match';
    }

    // 2. Optimization Model Result
    if (data.optimization) {
      renderGrid(data.optimization.prediction, 'opt-pred-grid', 'large');
      const isMatch = data.optimization.correct;
      const matchBadge = document.getElementById('opt-match-badge');
      matchBadge.textContent = isMatch ? '✓ EXACT MATCH (100%)' : '✗ OVERFIT DEMOS';
      matchBadge.className = `stat-badge ${isMatch ? 'match' : 'mismatch'}`;

      document.getElementById('opt-latency').textContent = `${data.optimization.latency_ms.toFixed(2)} ms`;
      document.getElementById('opt-param-change').textContent = `‖Δθ‖ = ${data.optimization.param_change_magnitude.toFixed(5)}`;

      const losses = data.optimization.loss_curve;
      const finalLoss = losses && losses.length > 0 ? losses[losses.length - 1].toFixed(4) : 'N/A';
      document.getElementById('opt-final-loss').textContent = finalLoss;

      // Table update
      document.getElementById('table-opt-lat').textContent = `${data.optimization.latency_ms.toFixed(2)} ms`;
      document.getElementById('table-opt-exact').textContent = isMatch ? '100% (Exact)' : '0.0% (Cell Overfit)';
    }

  } catch (err) {
    console.error('Comparison error:', err);
  } finally {
    btn.disabled = false;
    btn.innerHTML = '<span>⚡ Compare Models</span>';
  }
}

// State Convergence Trace
async function runStateTrace() {
  if (!currentPuzzle) return;

  try {
    const res = await fetch('/api/predict/context', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        puzzle: currentPuzzle,
        num_demos: 5
      })
    });
    const data = await res.json();
    const result = data.result;

    if (!result || !result.state_deltas) return;

    // Render Delta Bars
    const barsContainer = document.getElementById('delta-bars');
    barsContainer.innerHTML = '';

    const maxDelta = Math.max(...result.state_deltas.map(d => d.delta_norm), 1.0);

    result.state_deltas.forEach(d => {
      const row = document.createElement('div');
      row.className = 'delta-bar-row';

      const label = document.createElement('div');
      label.className = 'delta-bar-label';
      label.innerHTML = `<span>Ingest Demo ${d.to_demo} (from Demo ${d.from_demo}):</span> <strong>‖Δh‖ = ${d.delta_norm.toFixed(4)}</strong>`;

      const track = document.createElement('div');
      track.className = 'bar-track';

      const fill = document.createElement('div');
      fill.className = 'bar-fill';
      const pct = Math.min(100, (d.delta_norm / maxDelta) * 100);
      fill.style.width = `${pct}%`;

      track.appendChild(fill);
      row.appendChild(label);
      row.appendChild(track);
      barsContainer.appendChild(row);
    });

    // Render Sequential Demos Timeline
    const timeline = document.getElementById('snapshots-timeline');
    timeline.innerHTML = '';

    result.per_demo_states.forEach((snap, idx) => {
      const item = document.createElement('div');
      item.className = 'snapshot-item';

      const info = document.createElement('div');
      info.innerHTML = `<strong>Step ${idx + 1}: After Demo ${snap.demo_index + 1}</strong><br><small style="color:#94a3b8">State Norm ‖h‖ = ${snap.state_norm.toFixed(3)}</small>`;

      item.appendChild(info);
      timeline.appendChild(item);
    });

  } catch (err) {
    console.error('Failed to compute state trace:', err);
  }
}

// Technical Note Markdown Reader
async function loadTechNote() {
  try {
    const res = await fetch('/api/tech_note');
    const data = await res.json();
    const container = document.getElementById('tech-note-body');
    if (data.markdown) {
      container.textContent = data.markdown;
    }
  } catch (err) {
    console.error('Failed to load tech note:', err);
  }
}
