/**
 * app.js
 * Main browser controller for FormCoach (MAIB AI 221 · Project 10).
 * Coordinates camera/media input, MediaPipe Pose Landmarker, One-Euro smoothing,
 * rule-based & ONNX form evaluation, skeleton visualization, voice cues, and PDF reports.
 */

import { PoseEngine, POSE_CONNECTIONS } from './poseEngine.js';
import { PoseSmoother } from './smoothing.js';
import { extractBiomechanics } from './geometry.js';
import { ExerciseTracker } from './exerciseLogic.js';
import { OnnxFormClassifier } from './onnxClassifier.js';
import { VoiceCoach } from './voiceCoach.js';

// DOM Elements
const videoEl = document.getElementById('webcam-video');
const imgEl = document.getElementById('uploaded-image');
const canvas = document.getElementById('pose-canvas');
const ctx = canvas.getContext('2d');

const emptyState = document.getElementById('empty-state');
const hudTelemetry = document.getElementById('hud-telemetry');
const hudFps = document.getElementById('hud-fps');
const hudLatency = document.getElementById('hud-latency');
const hudConfidence = document.getElementById('hud-confidence');
const notSureBanner = document.getElementById('not-sure-banner');
const notSureReason = document.getElementById('not-sure-reason');
const cueBanner = document.getElementById('cue-banner');
const cueIcon = document.getElementById('cue-icon');
const cueText = document.getElementById('cue-text');

const exerciseSelect = document.getElementById('exercise-select');
const engineSelect = document.getElementById('engine-select');
const toggleVoice = document.getElementById('toggle-voice');
const toggleSmoothing = document.getElementById('toggle-smoothing');
const toggleAngles = document.getElementById('toggle-angles');

const btnStartCamHero = document.getElementById('btn-start-cam-hero');
const btnToggleCam = document.getElementById('btn-toggle-cam');
const btnFlipCam = document.getElementById('btn-flip-cam');
const fileInputHero = document.getElementById('file-input-hero');
const fileInput = document.getElementById('file-input');
const btnReset = document.getElementById('btn-reset');
const btnSummary = document.getElementById('btn-summary');

// Sidebar metrics
const statReps = document.getElementById('stat-reps');
const statGoodPct = document.getElementById('stat-good-pct');
const statPhase = document.getElementById('stat-phase');
const statTopIssue = document.getElementById('stat-top-issue');
const activeSideBadge = document.getElementById('active-side-badge');
const angleKneeEl = document.getElementById('angle-knee');
const angleHipEl = document.getElementById('angle-hip');
const angleTorsoEl = document.getElementById('angle-torso');
const angleElbowEl = document.getElementById('angle-elbow');
const onnxStatusEl = document.getElementById('onnx-status');
const repLogEl = document.getElementById('rep-log');
const repCountSubEl = document.getElementById('rep-count-sub');

// ONNX probability bars
const barGood = document.getElementById('bar-good');
const barDepth = document.getElementById('bar-depth');
const barAlign = document.getElementById('bar-align');
const valGood = document.getElementById('val-good');
const valDepth = document.getElementById('val-depth');
const valAlign = document.getElementById('val-align');

// Modals
const summaryModal = document.getElementById('summary-modal');
const helpModal = document.getElementById('help-modal');

// Core Modules
const poseEngine = new PoseEngine();
const smoother = new PoseSmoother(33, 1.4, 0.03);
const tracker = new ExerciseTracker('squat');
const onnxClassifier = new OnnxFormClassifier();
const voiceCoach = new VoiceCoach();

// Runtime State
let mode = 'idle'; // 'idle' | 'webcam' | 'video' | 'image'
let stream = null;
let facingMode = 'user';
let rafId = null;
let lastVideoTime = -1;
let frameTimes = [];
let onnxBusy = false;
let latestOnnxResult = null;
let isInitializingPose = false;

// --- Initialization ---
async function initModels() {
  onnxStatusEl.textContent = 'Loading ONNX...';
  const ok = await onnxClassifier.init();
  onnxStatusEl.textContent = ok ? 'ONNX Ready' : 'Rules Fallback';
  onnxStatusEl.style.color = ok ? '#6ee7b7' : '#fcd34d';
}

async function ensurePoseEngine(targetMode = 'VIDEO') {
  if (!poseEngine.ready) {
    if (isInitializingPose) return false;
    isInitializingPose = true;
    showCue('Loading MediaPipe Pose model (runs locally)...', 'neutral');
    try {
      await poseEngine.init(targetMode);
    } catch (err) {
      showCue(`Model load error: ${err.message}`, 'bad');
      isInitializingPose = false;
      return false;
    }
    isInitializingPose = false;
  } else {
    await poseEngine.setMode(targetMode);
  }
  return true;
}

// --- Media Controls ---
async function startWebcam() {
  stopMedia();
  const ready = await ensurePoseEngine('VIDEO');
  if (!ready) return;

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode,
        width: { ideal: 960 },
        height: { ideal: 540 },
      },
      audio: false,
    });
    videoEl.srcObject = stream;
    videoEl.loop = false;
    await videoEl.play();

    mode = 'webcam';
    emptyState.classList.add('hidden');
    hudTelemetry.classList.remove('hidden');
    cueBanner.classList.remove('hidden');
    btnToggleCam.textContent = 'Stop Webcam';
    btnToggleCam.classList.replace('btn-primary', 'btn-danger');
    btnFlipCam.classList.remove('hidden');

    canvas.width = videoEl.videoWidth || 960;
    canvas.height = videoEl.videoHeight || 540;
    smoother.reset();
    lastVideoTime = -1;

    requestAnimationFrame(processVideoLoop);
  } catch (err) {
    showCue(`Camera access denied or unavailable: ${err.message}`, 'bad');
  }
}

function stopMedia() {
  if (rafId) {
    cancelAnimationFrame(rafId);
    rafId = null;
  }
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }
  videoEl.pause();
  videoEl.srcObject = null;
  if (videoEl.src && videoEl.src.startsWith('blob:')) {
    URL.revokeObjectURL(videoEl.src);
  }
  videoEl.removeAttribute('src');
  mode = 'idle';

  btnToggleCam.textContent = 'Start Webcam';
  btnToggleCam.classList.replace('btn-danger', 'btn-primary');
  btnFlipCam.classList.add('hidden');
}

async function handleFileUpload(file) {
  if (!file) return;
  stopMedia();
  tracker.reset();
  smoother.reset();
  updateSidebarStats();

  const url = URL.createObjectURL(file);

  if (file.type.startsWith('image/')) {
    const ready = await ensurePoseEngine('IMAGE');
    if (!ready) return;

    imgEl.onload = async () => {
      mode = 'image';
      emptyState.classList.add('hidden');
      hudTelemetry.classList.remove('hidden');
      cueBanner.classList.remove('hidden');
      canvas.width = imgEl.naturalWidth || 960;
      canvas.height = imgEl.naturalHeight || 540;

      const t0 = performance.now();
      const rawLm = poseEngine.detectImage(imgEl);
      const elapsed = performance.now() - t0;

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(imgEl, 0, 0, canvas.width, canvas.height);

      hudFps.textContent = 'STATIC';
      hudLatency.textContent = `${elapsed.toFixed(1)} ms`;

      if (!rawLm) {
        showNotSure(true, 'No person detected in this image.');
        showCue('No pose detected', 'warn');
        hudConfidence.textContent = 'Pose: 0%';
        return;
      }

      const bio = extractBiomechanics(rawLm, tracker.exercise);
      const onnxRes = onnxClassifier.ready ? await onnxClassifier.predict(bio.featureVector) : null;
      if (onnxRes) updateOnnxBars(onnxRes);

      const state = tracker.evaluateStaticImage(bio, onnxRes, engineSelect.value);
      drawSkeleton(rawLm, bio, state.highlightJoints, false);
      updateLiveTelemetry(bio, state);
      URL.revokeObjectURL(url);
    };
    imgEl.src = url;
  } else if (file.type.startsWith('video/')) {
    const ready = await ensurePoseEngine('VIDEO');
    if (!ready) return;

    videoEl.src = url;
    videoEl.loop = true;
    videoEl.onloadeddata = async () => {
      mode = 'video';
      emptyState.classList.add('hidden');
      hudTelemetry.classList.remove('hidden');
      cueBanner.classList.remove('hidden');
      canvas.width = videoEl.videoWidth || 960;
      canvas.height = videoEl.videoHeight || 540;
      lastVideoTime = -1;
      await videoEl.play();
      requestAnimationFrame(processVideoLoop);
    };
  }
}

// --- Real-Time Processing Loop ---
function processVideoLoop() {
  if (mode !== 'webcam' && mode !== 'video') return;

  if (videoEl.readyState >= 2 && videoEl.currentTime !== lastVideoTime) {
    lastVideoTime = videoEl.currentTime;
    const now = performance.now();

    // FPS rolling window (last 20 frames)
    frameTimes.push(now);
    if (frameTimes.length > 20) frameTimes.shift();
    if (frameTimes.length >= 2) {
      const fps = ((frameTimes.length - 1) * 1000) / (frameTimes[frameTimes.length - 1] - frameTimes[0]);
      hudFps.textContent = `${Math.round(fps)} FPS`;
    }

    const t0 = performance.now();
    const rawLm = poseEngine.detectVideo(videoEl, now);
    const inferMs = performance.now() - t0;
    hudLatency.textContent = `${inferMs.toFixed(1)} ms`;

    // Draw video frame (mirror webcam for natural user feedback)
    const mirror = mode === 'webcam' && facingMode === 'user';
    ctx.save();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (mirror) {
      ctx.translate(canvas.width, 0);
      ctx.scale(-1, 1);
    }
    ctx.drawImage(videoEl, 0, 0, canvas.width, canvas.height);
    ctx.restore();

    if (rawLm) {
      const landmarks = toggleSmoothing.checked ? smoother.smooth(rawLm, now) : rawLm;
      const bio = extractBiomechanics(landmarks, tracker.exercise);

      // Run ONNX classifier asynchronously without blocking 30fps render loop
      if (onnxClassifier.ready && !onnxBusy) {
        onnxBusy = true;
        onnxClassifier
          .predict(bio.featureVector)
          .then((res) => {
            latestOnnxResult = res;
            if (res) updateOnnxBars(res);
          })
          .finally(() => {
            onnxBusy = false;
          });
      }

      const state = tracker.update(bio, now, latestOnnxResult, engineSelect.value);
      drawSkeleton(landmarks, bio, state.highlightJoints, mirror);
      updateLiveTelemetry(bio, state);

      if (state.repCompleted) {
        appendRepLog(state.repCompleted);
        updateSidebarStats();
      }
      if (state.cue?.speak) {
        voiceCoach.speak(state.cue.speak, Boolean(state.repCompleted));
      }
    } else {
      showNotSure(true, 'No person detected in frame. Step into view.');
      hudConfidence.textContent = 'Pose: 0%';
      hudConfidence.className = 'hud-badge conf-low';
    }
  }

  rafId = requestAnimationFrame(processVideoLoop);
}

// --- Canvas Skeleton & Angle Renderer ---
function drawSkeleton(landmarks, bio, highlightJoints = [], mirror = false) {
  const w = canvas.width;
  const h = canvas.height;
  const toX = (nx) => (mirror ? (1 - nx) * w : nx * w);
  const toY = (ny) => ny * h;

  const isIssue = highlightJoints.length > 0;
  const lineColor = isIssue ? '#f59e0b' : '#10b981';

  // Bones
  ctx.lineWidth = Math.max(3, w * 0.004);
  ctx.lineCap = 'round';
  for (const [a, b] of POSE_CONNECTIONS) {
    const pa = landmarks[a];
    const pb = landmarks[b];
    if (!pa || !pb || (pa.visibility ?? 0) < 0.4 || (pb.visibility ?? 0) < 0.4) continue;
    ctx.strokeStyle = lineColor;
    ctx.beginPath();
    ctx.moveTo(toX(pa.x), toY(pa.y));
    ctx.lineTo(toX(pb.x), toY(pb.y));
    ctx.stroke();
  }

  // Joints
  const highlightPoints = new Set(highlightJoints.map((name) => bio.joints[name]).filter(Boolean));
  for (let i = 11; i < landmarks.length; i++) {
    const p = landmarks[i];
    if (!p || (p.visibility ?? 0) < 0.4) continue;
    const isBadJoint = highlightPoints.has(p);
    ctx.fillStyle = isBadJoint ? '#ef4444' : '#f8fafc';
    ctx.strokeStyle = isBadJoint ? '#fecaca' : '#0f172a';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(toX(p.x), toY(p.y), isBadJoint ? 7 : 4.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  }

  // Optional live angle tags on key joints
  if (toggleAngles.checked) {
    const drawTag = (pt, text) => {
      if (!pt || (pt.visibility ?? 0) < 0.45) return;
      const x = toX(pt.x) + 10;
      const y = toY(pt.y) - 8;
      ctx.font = 'bold 13px Inter, sans-serif';
      const tw = ctx.measureText(text).width;
      ctx.fillStyle = 'rgba(15, 23, 42, 0.82)';
      ctx.fillRect(x - 4, y - 13, tw + 8, 18);
      ctx.fillStyle = '#f8fafc';
      ctx.fillText(text, x, y);
    };

    if (tracker.exercise === 'pushup') {
      drawTag(bio.joints.elbow, `${Math.round(bio.angles.elbowAngle)}°`);
      drawTag(bio.joints.hip, `${Math.round(bio.angles.bodyLineAngle)}°`);
    } else {
      drawTag(bio.joints.knee, `${Math.round(bio.angles.kneeAngle)}°`);
      drawTag(bio.joints.hip, `${Math.round(bio.angles.hipAngle)}°`);
    }
  }
}

// --- UI Helpers ---
function updateLiveTelemetry(bio, state) {
  const confPct = Math.round(bio.avgKeyVis * 100);
  hudConfidence.textContent = `Pose: ${confPct}%`;
  hudConfidence.className = `hud-badge ${confPct >= 75 ? 'conf-high' : confPct >= 60 ? 'conf-med' : 'conf-low'}`;

  activeSideBadge.textContent = `${bio.side.toUpperCase()} Side`;
  angleKneeEl.textContent = `${Math.round(bio.angles.kneeAngle)}°`;
  angleHipEl.textContent = `${Math.round(bio.angles.hipAngle)}°`;
  angleTorsoEl.textContent = `${Math.round(bio.angles.torsoTiltDeg)}°`;
  angleElbowEl.textContent = `${Math.round(bio.angles.elbowAngle)}°`;
  statPhase.textContent = state.phase;

  showNotSure(state.isNotSure, state.notSureReason);
  if (state.cue) {
    showCue(state.cue.text, state.cue.level);
  }
}

function updateOnnxBars(res) {
  const [pGood, pDepth, pAlign] = res.probs.map((p) => Math.round(p * 100));
  barGood.style.width = `${pGood}%`;
  barDepth.style.width = `${pDepth}%`;
  barAlign.style.width = `${pAlign}%`;
  valGood.textContent = `${pGood}%`;
  valDepth.textContent = `${pDepth}%`;
  valAlign.textContent = `${pAlign}%`;
}

function showNotSure(visible, reason = '') {
  if (visible) {
    notSureReason.textContent = reason;
    notSureBanner.classList.remove('hidden');
  } else {
    notSureBanner.classList.add('hidden');
  }
}

function showCue(text, level = 'neutral') {
  const icons = { good: '✅', warn: '⚠️', bad: '🛑', neutral: 'ℹ️' };
  cueIcon.textContent = icons[level] || 'ℹ️';
  cueText.textContent = text;
  cueBanner.className = `cue-banner cue-${level}`;
}

function updateSidebarStats() {
  const s = tracker.getSummary();
  statReps.textContent = String(s.totalReps);
  statGoodPct.textContent = s.totalReps > 0 ? `${s.goodPct}%` : '--%';
  statTopIssue.textContent = s.topMistake;
  repCountSubEl.textContent = `${s.totalReps} logged`;
}

function appendRepLog(rep) {
  const emptyItem = repLogEl.querySelector('.rep-empty');
  if (emptyItem) emptyItem.remove();

  const li = document.createElement('li');
  li.className = `rep-item ${rep.isGood ? '' : 'rep-bad'}`;
  li.innerHTML = `
    <span><strong>Rep #${rep.repNumber}</strong> · ${rep.feedback}</span>
    <span class="mini-badge">${rep.minAngle}°</span>
  `;
  repLogEl.prepend(li);
}

function openSummaryModal() {
  const s = tracker.getSummary();
  document.getElementById('sum-exercise').textContent = s.exerciseName;
  document.getElementById('sum-reps').textContent = String(s.totalReps);
  document.getElementById('sum-good-pct').textContent = s.totalReps > 0 ? `${s.goodPct}%` : '--%';
  document.getElementById('sum-mistake').textContent = s.topMistake;
  document.getElementById('print-timestamp').textContent = `Generated on ${new Date().toLocaleString()} · Engine: ${engineSelect.options[engineSelect.selectedIndex].text}`;

  const tbody = document.getElementById('summary-table-body');
  if (s.history.length === 0) {
    tbody.innerHTML = '<tr><td colspan="5">No reps recorded yet. Complete a session or upload a workout clip.</td></tr>';
  } else {
    tbody.innerHTML = s.history
      .map(
        (r) => `
      <tr>
        <td>#${r.repNumber}</td>
        <td>${r.isGood ? '✅ Good' : '⚠️ Needs Form Fix'}</td>
        <td>${r.minAngle}°</td>
        <td>${r.feedback}</td>
        <td>${r.confidence}%</td>
      </tr>`
      )
      .join('');
  }

  summaryModal.classList.remove('hidden');
}

// --- Event Listeners ---
btnStartCamHero.addEventListener('click', startWebcam);
btnToggleCam.addEventListener('click', () => {
  if (mode === 'webcam') {
    stopMedia();
    emptyState.classList.remove('hidden');
    hudTelemetry.classList.add('hidden');
    cueBanner.classList.add('hidden');
    showNotSure(false);
  } else {
    startWebcam();
  }
});

btnFlipCam.addEventListener('click', () => {
  facingMode = facingMode === 'user' ? 'environment' : 'user';
  startWebcam();
});

fileInputHero.addEventListener('change', (e) => handleFileUpload(e.target.files?.[0]));
fileInput.addEventListener('change', (e) => handleFileUpload(e.target.files?.[0]));

exerciseSelect.addEventListener('change', () => {
  tracker.setExercise(exerciseSelect.value);
  smoother.reset();
  repLogEl.innerHTML = '<li class="rep-empty">Complete a rep to see form feedback here.</li>';
  updateSidebarStats();
  showCue(`Switched to ${tracker.config.name}s`, 'neutral');
});

toggleVoice.addEventListener('change', () => {
  voiceCoach.setEnabled(toggleVoice.checked);
  if (toggleVoice.checked) {
    voiceCoach.speak('Voice coaching enabled', true);
  }
});

btnReset.addEventListener('click', () => {
  tracker.reset();
  smoother.reset();
  repLogEl.innerHTML = '<li class="rep-empty">Complete a rep to see form feedback here.</li>';
  updateSidebarStats();
  showCue('Session reset', 'neutral');
});

btnSummary.addEventListener('click', openSummaryModal);
document.getElementById('btn-close-summary').addEventListener('click', () => summaryModal.classList.add('hidden'));
document.getElementById('btn-close-summary-2').addEventListener('click', () => summaryModal.classList.add('hidden'));
document.getElementById('btn-print-pdf').addEventListener('click', () => window.print());

document.getElementById('btn-help').addEventListener('click', () => helpModal.classList.remove('hidden'));
document.getElementById('btn-close-help').addEventListener('click', () => helpModal.classList.add('hidden'));
document.getElementById('btn-close-help-2').addEventListener('click', () => helpModal.classList.add('hidden'));

// Boot ONNX classifier on load
initModels();
