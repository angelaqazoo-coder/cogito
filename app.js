/**
 * Cogito — app.js
 * Frontend logic: WebSocket session, mic capture via AudioWorklet,
 * Gemini audio playback, KaTeX workspace rendering, SVG annotations, flowchart.
 */
'use strict';

/* ── Config ──────────────────────────────────────────────────────────────── */
const WS_URL = (() => {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const host  = location.host || 'localhost:8000';
  return `${proto}://${host}/ws`;
})();

const SESSION_ID = crypto.randomUUID();
const MIC_SAMPLE_RATE = 16000;   // Required by Gemini Live
const AI_SAMPLE_RATE  = 24000;   // Output from Gemini Live

/* ── DOM refs ────────────────────────────────────────────────────────────── */
const $  = id => document.getElementById(id);
const elConnStatus    = $('connection-status');
const elStatusLabel   = elConnStatus.querySelector('.status-label');
const elClock         = $('session-clock');
const elDropZone      = $('drop-zone');
const elFileInput     = $('file-input');
const elUploadState   = $('upload-state');
const elProblemLoaded = $('problem-loaded');
const elProblemImg    = $('problem-img');
const elAnnotationSvg = $('annotation-svg');
const elAnnTooltip    = $('annotation-tooltip');
const elAnnTipText    = $('annotation-tooltip-text');
const elProblemSummary= $('problem-summary');
const elRemoveImg     = $('remove-img-btn');
const elTopicChips    = $('problem-topics');
const elWorkspaceEmpty= $('workspace-empty');
const elCurrentStep   = $('current-step');
const elStepBadge     = $('step-status-badge');
const elStepLabel     = $('step-label-disp');
const elLatexBox      = $('latex-display');
const elCorrectionBox = $('correction-box');
const elLatexWrong    = $('latex-wrong');
const elLatexCorrect  = $('latex-correct');
const elHintCard      = $('hint-card');
const elHintIcon      = $('hint-icon');
const elHintTypeLabel = $('hint-type-label');
const elHintText      = $('hint-text-content');
const elFlowSection   = $('flowchart-section');
const elFlowSvg       = $('flowchart-svg');
const elMicBtn        = $('mic-btn');
const elWaveform      = $('waveform-display');
const elWaveIdle      = $('wave-idle-label');
const elVolSlider     = $('vol-slider');
const elMicHint       = $('mic-hint');

/* ── State ───────────────────────────────────────────────────────────────── */
const state = {
  connected:   false,
  recording:   false,
  aiSpeaking:  false,
  timerSec:    0,
  timerHandle: null,
  imageLoaded: false,
  imageData:   null,   // base64 string
  imageMime:   'image/png',
  annotations:  [],    // from /api/analyze
  flowNodes:    [],    // {label, status}
  gainNode:     null,
};

/* ── WebSocket ───────────────────────────────────────────────────────────── */
let ws = null;

function connectWS() {
  const url = `${WS_URL}/${SESSION_ID}`;
  ws = new WebSocket(url);
  ws.binaryType = 'arraybuffer';

  ws.addEventListener('open', () => {
    state.connected = true;
    setStatus('online', 'Cogito · Live');
    startTimer();
    // If image already loaded, send init
    if (state.imageData) sendInit();
  });

  ws.addEventListener('message', onWsMessage);

  ws.addEventListener('close', () => {
    setStatus('offline', 'Offline');
    stopMic();
    // Attempt reconnect after 3 s
    setTimeout(connectWS, 3000);
  });

  ws.addEventListener('error', () => {
    setStatus('error', 'Error');
  });
}

function sendJSON(obj) {
  if (ws?.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(obj));
  }
}

function sendInit() {
  sendJSON({ type: 'init', image: state.imageData, mime: state.imageMime });
}

/* ── WS incoming messages ────────────────────────────────────────────────── */
function onWsMessage(evt) {
  if (typeof evt.data === 'string') {
    const msg = JSON.parse(evt.data);
    if (msg.type === 'audio')            handleAudioChunk(msg.data, msg.mime);
    else if (msg.type === 'workspace_update') handleWorkspaceUpdate(msg.payload);
    else if (msg.type === 'turn_complete')    handleTurnComplete();
    else if (msg.type === 'error')       console.error('Server error:', msg.message);
  }
}

/* ── Session timer ───────────────────────────────────────────────────────── */
function startTimer() {
  if (state.timerHandle) return;
  state.timerHandle = setInterval(() => {
    state.timerSec++;
    const m = String(Math.floor(state.timerSec / 60)).padStart(2,'0');
    const s = String(state.timerSec % 60).padStart(2,'0');
    elClock.textContent = `${m}:${s}`;
  }, 1000);
}

/* ── Status pill ─────────────────────────────────────────────────────────── */
function setStatus(cls, label) {
  elConnStatus.className = `status-pill status-${cls}`;
  elStatusLabel.textContent = label;
}

/* ── Image upload ────────────────────────────────────────────────────────── */
elFileInput.addEventListener('change', () => handleFile(elFileInput.files[0]));
elDropZone.addEventListener('dragover',  e => { e.preventDefault(); elDropZone.classList.add('drag-over'); });
elDropZone.addEventListener('dragleave', () => elDropZone.classList.remove('drag-over'));
elDropZone.addEventListener('drop', e => { e.preventDefault(); elDropZone.classList.remove('drag-over'); handleFile(e.dataTransfer.files[0]); });
elDropZone.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') elFileInput.click(); });
document.addEventListener('paste', e => {
  const img = Array.from(e.clipboardData.items).find(i => i.type.startsWith('image/'));
  if (img) handleFile(img.getAsFile());
});
elRemoveImg.addEventListener('click', resetProblem);

async function handleFile(file) {
  if (!file || !file.type.startsWith('image/')) return;

  const reader = new FileReader();
  reader.onload = async e => {
    const dataUrl  = e.target.result;
    const base64   = dataUrl.split(',')[1];

    state.imageData = base64;
    state.imageMime = file.type;
    state.imageLoaded = true;

    // Show image
    elProblemImg.src = dataUrl;
    elUploadState.classList.add('hidden');
    elProblemLoaded.classList.remove('hidden');

    // If connected, send init immediately
    if (state.connected) sendInit();

    // Analyse image
    await analyseImage(file);
  };
  reader.readAsDataURL(file);
}

async function analyseImage(file) {
  const form = new FormData();
  form.append('file', file);
  try {
    const res  = await fetch('/api/analyze', { method: 'POST', body: form });
    const data = await res.json();
    applyAnalysis(data);
  } catch (err) {
    console.warn('Image analysis unavailable:', err);
    elProblemSummary.textContent = 'Problem loaded. Speak to begin.';
  }
}

function applyAnalysis(data) {
  elProblemSummary.textContent = data.problem_summary || '';

  // Topic chips
  elTopicChips.innerHTML = '';
  (data.topics || []).forEach((t, i) => {
    const chip = document.createElement('span');
    chip.className = `chip${i === 0 ? ' active' : ''}`;
    chip.textContent = t;
    elTopicChips.appendChild(chip);
  });

  // Annotations
  state.annotations = data.annotations || [];
  renderAnnotations();
}

/* ── SVG Annotation Overlay ──────────────────────────────────────────────── */
const ANN_FILL_COLORS = {
  definition: '#3b82f6',
  constraint: '#f97316',
  assumption:  '#a855f7',
  unknown:     '#eab308',
};
const ANN_ICONS = {
  definition: '🔵',
  constraint: '🟠',
  assumption:  '🟣',
  unknown:     '🟡',
};
const ANN_LABELS = {
  definition: 'Definition',
  constraint: 'Constraint',
  assumption:  'Assumption',
  unknown:     '?',
};

function renderAnnotations() {
  elAnnotationSvg.innerHTML = '';
  // Active region ring (for current step highlight)
  const ring = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
  ring.setAttribute('id', 'ann-active-ring');
  ring.setAttribute('class', 'ann-active-ring');
  ring.setAttribute('rx', 6);
  elAnnotationSvg.appendChild(ring);

  state.annotations.forEach((ann, i) => {
    if (!ann.box_2d || ann.box_2d.length < 4) return;
    const [y1, x1, y2, x2] = ann.box_2d;
    const colour = ANN_FILL_COLORS[ann.type] || '#6C63FF';

    const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    g.setAttribute('class', `ann-group ann-group-${i}`);

    // Rect
    const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    rect.setAttribute('x', x1); rect.setAttribute('y', y1);
    rect.setAttribute('width', x2 - x1); rect.setAttribute('height', y2 - y1);
    rect.setAttribute('rx', 6);
    rect.setAttribute('class', `ann-rect ann-${ann.type}`);
    rect.setAttribute('stroke', colour);
    rect.setAttribute('fill', colour.replace(')', ',0.08)').replace('rgb', 'rgba'));
    rect.setAttribute('stroke-width', 2);
    rect.setAttribute('fill-opacity', 0.1);
    g.appendChild(rect);

    // Label background
    const labelBg = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    const labelText = `${ANN_LABELS[ann.type] || ann.type}: ${ann.text}`;
    const labelW = Math.min(labelText.length * 5.5 + 12, 200);
    labelBg.setAttribute('x', x1); labelBg.setAttribute('y', Math.max(0, y1 - 20));
    labelBg.setAttribute('width', labelW); labelBg.setAttribute('height', 18);
    labelBg.setAttribute('rx', 4);
    labelBg.setAttribute('fill', colour);
    labelBg.setAttribute('class', 'ann-label-bg');
    g.appendChild(labelBg);

    const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    text.setAttribute('x', x1 + 6); text.setAttribute('y', Math.max(0, y1 - 11));
    text.setAttribute('class', 'ann-label-text');
    text.setAttribute('fill', '#fff');
    text.setAttribute('font-size', '11');
    text.setAttribute('font-weight', '600');
    text.textContent = `${ANN_ICONS[ann.type] || ''} ${labelText}`;
    g.appendChild(text);

    // Hover tooltip
    g.addEventListener('mouseenter', () => {
      elAnnTipText.textContent = `${ANN_LABELS[ann.type]}: ${ann.text}`;
      elAnnTooltip.classList.remove('hidden');
    });
    g.addEventListener('mouseleave', () => elAnnTooltip.classList.add('hidden'));

    // Fade in
    g.style.opacity = 0;
    elAnnotationSvg.appendChild(g);
    requestAnimationFrame(() => {
      g.style.transition = `opacity 0.4s ease ${i * 0.12}s`;
      g.style.opacity    = 1;
    });
  });
}

function highlightAnnotationRegion(description) {
  // Try to find a matching annotation by keyword in its text
  if (!description) return clearAnnotationHighlight();
  const lower = description.toLowerCase();
  const match = state.annotations.find(a =>
    a.text.toLowerCase().includes(lower.split(' ').find(w => w.length > 3) || lower)
  );
  if (match && match.box_2d) {
    const [y1, x1, y2, x2] = match.box_2d;
    const ring = document.getElementById('ann-active-ring');
    if (ring) {
      ring.setAttribute('x', x1 - 4);  ring.setAttribute('y', y1 - 4);
      ring.setAttribute('width',  x2 - x1 + 8);
      ring.setAttribute('height', y2 - y1 + 8);
      ring.classList.add('visible');
    }
  } else {
    clearAnnotationHighlight();
  }
}

function clearAnnotationHighlight() {
  document.getElementById('ann-active-ring')?.classList.remove('visible');
}

function resetProblem() {
  state.imageLoaded = false;
  state.imageData   = null;
  state.annotations = [];
  elProblemImg.src  = '';
  elAnnotationSvg.innerHTML = '';
  elTopicChips.innerHTML    = '';
  elProblemSummary.textContent = '';
  elProblemLoaded.classList.add('hidden');
  elUploadState.classList.remove('hidden');
  elFileInput.value = '';
}

/* ── Workspace — KaTeX rendering ─────────────────────────────────────────── */
function renderKaTeX(el, latex) {
  try {
    if (window.katex) {
      katex.render(latex, el, { throwOnError: false, displayMode: true, strict: false });
    } else {
      el.textContent = latex;
    }
  } catch {
    el.textContent = latex;
  }
}

function handleWorkspaceUpdate(payload) {
  const { latex, status, step_label, hint_text, hint_type, corrected_latex, annotation_region } = payload;

  // ── Current step box ────────────────────────────────────────────────────
  elWorkspaceEmpty.classList.add('hidden');
  elCurrentStep.classList.remove('hidden', 'status-correct', 'status-error', 'status-partial');
  const cssStatus = status === 'correct' ? 'status-correct'
                  : status === 'error'   ? 'status-error'
                  :                        'status-partial';
  elCurrentStep.classList.add(cssStatus);

  const badgeText = status === 'correct' ? '✓ Correct' : status === 'error' ? '✗ Error' : '~ Partial';
  elStepBadge.textContent = badgeText;
  elStepLabel.textContent = step_label || '';

  // Trigger re-animation
  elCurrentStep.style.animation = 'none';
  void elCurrentStep.offsetHeight;
  elCurrentStep.style.animation = '';

  // Render LaTeX
  if (status === 'error' && corrected_latex) {
    // Show diff: wrong version vs corrected
    elLatexBox.classList.add('hidden');
    elCorrectionBox.classList.remove('hidden');
    renderKaTeX(elLatexWrong, latex);
    renderKaTeX(elLatexCorrect, corrected_latex);
  } else {
    elLatexBox.classList.remove('hidden');
    elCorrectionBox.classList.add('hidden');
    renderKaTeX(elLatexBox, latex);
  }

  // ── Hint card ───────────────────────────────────────────────────────────
  if (hint_text && hint_type) {
    const icons = { hint: '💡', fallacy: '⚠️', theory: '📖' };
    const labels = { hint: 'Hint', fallacy: 'Common Error', theory: 'Background' };
    elHintCard.className = `hint-type-${hint_type}`;
    elHintCard.classList.remove('hidden');
    elHintIcon.textContent = icons[hint_type] || '💡';
    elHintTypeLabel.textContent = labels[hint_type] || hint_type;
    elHintText.textContent = hint_text;
  } else {
    elHintCard.classList.add('hidden');
  }

  // ── Annotation highlight ─────────────────────────────────────────────────
  highlightAnnotationRegion(annotation_region);

  // ── Flowchart ────────────────────────────────────────────────────────────
  state.flowNodes.push({ label: step_label, status });
  elFlowSection.classList.remove('hidden');
  renderFlowchart();
}

function handleTurnComplete() {
  elWaveform.classList.remove('ai-speaking');
}

/* ── Flowchart SVG ───────────────────────────────────────────────────────── */
const FC = { nodeW: 240, nodeH: 46, gapY: 66, startX: 8, startY: 16 };

function renderFlowchart() {
  const totalH = FC.startY + state.flowNodes.length * FC.gapY + 24;
  elFlowSvg.innerHTML = '';
  elFlowSvg.setAttribute('viewBox', `0 0 260 ${totalH}`);
  elFlowSvg.setAttribute('height', totalH);

  state.flowNodes.forEach((node, i) => {
    const x = FC.startX;
    const y = FC.startY + i * FC.gapY;
    const cx = FC.startX + FC.nodeW / 2;

    // Connector
    if (i > 0) {
      const line = svgEl('line');
      attrs(line, { x1: cx, y1: y - FC.gapY + FC.nodeH, x2: cx, y2: y, class: 'fc-connector' });
      elFlowSvg.appendChild(line);
    }

    const statusClass = node.status === 'correct' ? 'fc-node-correct'
                       : node.status === 'error'  ? 'fc-node-error'
                       :                            'fc-node-partial';

    const g = svgEl('g');
    g.setAttribute('class', `fc-node-group ${statusClass}`);
    g.setAttribute('transform', `translate(${x},${y})`);
    g.style.opacity = 0;

    const rect = svgEl('rect');
    attrs(rect, { x: 0, y: 0, width: FC.nodeW, height: FC.nodeH, rx: 8 });
    g.appendChild(rect);

    const icon = { correct: '✓', error: '✗', partial: '~' }[node.status] || '·';
    const numT = svgText(`${i + 1}.`, 10, 24, 'fc-num');
    const lblT = svgText(node.label, 26, 22, 'fc-label');
    const icnT = svgText(icon, FC.nodeW - 14, 24, 'fc-label');
    icnT.setAttribute('text-anchor', 'middle');

    g.appendChild(numT);
    g.appendChild(lblT);
    g.appendChild(icnT);
    elFlowSvg.appendChild(g);

    requestAnimationFrame(() => { g.style.transition = 'opacity 0.4s ease'; g.style.opacity = 1; });
  });
}

function svgEl(tag) { return document.createElementNS('http://www.w3.org/2000/svg', tag); }
function attrs(el, map) { Object.entries(map).forEach(([k,v]) => el.setAttribute(k, v)); }
function svgText(txt, x, y, cls) {
  const t = svgEl('text');
  t.setAttribute('x', x); t.setAttribute('y', y);
  t.setAttribute('class', cls);
  t.setAttribute('dominant-baseline', 'middle');
  t.textContent = txt;
  return t;
}

/* ── Audio Contexts (separate rates for capture vs playback) ─────────────── */
let micCtx      = null;   // 16 kHz — mic AudioWorklet
let playCtx     = null;   // 24 kHz — AI audio output
let nextPlayTime = 0;
let gainNode     = null;

function ensureMicCtx() {
  if (!micCtx) {
    micCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: MIC_SAMPLE_RATE });
  }
  if (micCtx.state === 'suspended') micCtx.resume();
  return micCtx;
}

function ensurePlayCtx() {
  if (!playCtx) {
    playCtx  = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: AI_SAMPLE_RATE });
    gainNode = playCtx.createGain();
    gainNode.connect(playCtx.destination);
    gainNode.gain.value = parseFloat(elVolSlider.value);
    state.gainNode = gainNode;
  }
  if (playCtx.state === 'suspended') playCtx.resume();
  return playCtx;
}

elVolSlider.addEventListener('input', () => {
  if (state.gainNode) state.gainNode.gain.value = parseFloat(elVolSlider.value);
});

function handleAudioChunk(base64, mime) {
  const ctx = ensurePlayCtx();
  elWaveform.classList.add('ai-speaking');

  // Decode base64 → ArrayBuffer
  const binary = atob(base64);
  const bytes  = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);

  // PCM16 LE → Float32
  const pcm16   = new DataView(bytes.buffer);
  const samples = bytes.length / 2;
  const float32 = new Float32Array(samples);
  for (let i = 0; i < samples; i++) {
    float32[i] = pcm16.getInt16(i * 2, true) / 32768;
  }

  const audioBuf = ctx.createBuffer(1, float32.length, AI_SAMPLE_RATE);
  audioBuf.copyToChannel(float32, 0);

  const src = ctx.createBufferSource();
  src.buffer = audioBuf;
  src.connect(gainNode);

  const startAt = Math.max(nextPlayTime, ctx.currentTime + 0.01);
  src.start(startAt);
  nextPlayTime = startAt + audioBuf.duration;
}

/* ── Mic Capture ─────────────────────────────────────────────────────────── */
let micStream    = null;
let audioWorklet = null;
let workletNode  = null;
let waveInterval = null;

// Waveform bars
const BAR_N = 28;
const bars  = [];
for (let i = 0; i < BAR_N; i++) {
  const b = document.createElement('div');
  b.className = 'wave-bar';
  elWaveform.appendChild(b);
  bars.push(b);
}

function animateWave(active) {
  clearInterval(waveInterval);
  if (active) {
    elWaveform.classList.add('active');
    waveInterval = setInterval(() => {
      bars.forEach(b => { b.style.height = (5 + Math.random() * 28) + 'px'; });
    }, 80);
  } else {
    elWaveform.classList.remove('active', 'ai-speaking');
    bars.forEach(b => { b.style.height = '4px'; });
  }
}

elMicBtn.addEventListener('click', toggleMic);
document.addEventListener('keydown', e => {
  if (e.code === 'Space' && e.target === document.body) { e.preventDefault(); toggleMic(); }
});

async function toggleMic() {
  if (state.recording) { stopMic(); }
  else                 { await startMic(); }
}

async function startMic() {
  try {
    const ctx = ensureMicCtx();
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
    });

    await ctx.audioWorklet.addModule('audio-processor.js');
    const source = ctx.createMediaStreamSource(micStream);
    workletNode  = new AudioWorkletNode(ctx, 'pcm-processor');

    workletNode.port.onmessage = e => {
      if (ws?.readyState === WebSocket.OPEN) {
        ws.send(e.data); // send ArrayBuffer (PCM16) directly as binary WS frame
      }
    };

    source.connect(workletNode);
    // Do NOT connect workletNode to destination (no mic feedback)

    state.recording = true;
    elMicBtn.classList.add('recording');
    elMicBtn.setAttribute('aria-pressed', 'true');
    animateWave(true);
  } catch (err) {
    console.error('Mic error:', err);
    alert('Microphone access denied or unavailable.');
  }
}

function stopMic() {
  workletNode?.disconnect();
  workletNode = null;
  micStream?.getTracks().forEach(t => t.stop());
  micStream = null;
  state.recording = false;
  elMicBtn.classList.remove('recording');
  elMicBtn.setAttribute('aria-pressed', 'false');
  animateWave(false);
}

/* ── Boot ────────────────────────────────────────────────────────────────── */
connectWS();

// Expose for debugging
window._cogito = { state, ws, sendJSON };
