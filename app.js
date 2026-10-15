/* =====================================================
   DrowseGuard – Driver Drowsiness Detection Engine
   ===================================================== */

// ── Constants ─────────────────────────────────────────
const EAR_THRESHOLD       = 0.25;   // below this = eyes closing
const CONSEC_FRAMES_ALERT = 15;     // frames of low EAR before alert
const MEDIAPIPE_VERSION   = '1.1.0';
const MEDIAPIPE_CDN       = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}`;
const MEDIAPIPE_MODEL_URL  = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';
const PERCLOS_WINDOW_MS    = 60_000;
const LOG_INTERVAL_MS      = 250;
const NO_FACE_ALERT_MS     = 15_000;
const GAZE_AWAY_ALERT_MS   = 1_200;

// Sensitivity presets: [frame threshold multiplier, EAR threshold]
const SENSITIVITY = {
  1: { frames: 25, alertMs: 1500, ear: 0.22, label: 'Low' },
  2: { frames: 15, alertMs: 1000, ear: 0.25, label: 'Medium' },
  3: { frames: 8,  alertMs: 700,  ear: 0.28, label: 'High' },
};

// ── State ─────────────────────────────────────────────
let isRunning        = false;
let animFrameId      = null;
let stream           = null;
let sessionStart     = null;
let sessionTimer     = null;
let earBuffer        = [];          // rolling buffer for smoothing
let closedFrames     = 0;
let blinkCount       = 0;
let longClosureCount = 0;
let alertCount       = 0;
let eyeClosedStart   = null;
let wasEyeClosed     = false;
let drowsinessScore  = 0;           // 0–100
let lastAlertTime    = 0;
let faceMissingStart = null;
let faceLossAlerted  = false;
let blinkTrackingClosed = false;
let blinkTrackingStart = null;
let gazeAwayStart = null;
let gazeAwayAlerted = false;
let gazeAlignmentBuffer = [];
let currentHeadAlignment = 100;
let gazeTurnDirection = 'Frontal';
const HEAD_ALIGNMENT_THRESHOLD = 70; // 70% minimum visibility restriction from faculty
let fpsFrames        = 0;
let fpsLastTime      = performance.now();
let currentFps       = 0;
let sens             = SENSITIVITY[2];
let modelsLoaded     = false;
let mediaPipeLandmarker = null;
let selectedModel    = 'mediapipe';
let detectionRunId   = 0;
let perclosSamples   = [];
let lastLoggedAt     = 0;
let lastInferenceMs  = 0;
let lastScoreUpdate  = 0;
let evaluation = { frames: 0, faceFrames: 0, earTotal: 0 };
let logs             = [];
let earHistory       = [];
let voiceSettings    = { enabled: false, volume: 0.8, level: 3 };
let voiceTimer       = null;
let currentVoiceLevel = 0;
const voiceMessages   = {
  1: 'Please stay alert.',
  2: 'Drowsiness detected. Please take a break.',
  3: 'Warning. Pull over safely and rest.'
};
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');

function openSettings() {
  if (document.getElementById('settingsModal')) return;
  const modal = document.createElement('div');
  modal.id = 'settingsModal';
  modal.className = 'settings-modal';
  modal.innerHTML = `
    <div class="settings-card" role="dialog" aria-modal="true" aria-labelledby="settingsTitle">
      <h3 id="settingsTitle">Detection Settings</h3>
      <label for="settingsSensitivity">Alert sensitivity</label>
      <select id="settingsSensitivity">
        <option value="1">Low</option>
        <option value="2">Medium</option>
        <option value="3">High</option>
      </select>
      <label for="settingsEar">EAR threshold</label>
      <input id="settingsEar" type="number" min="0.1" max="0.5" step="0.01" value="${sens.ear}">
      <label><input id="settingsVoiceEnabled" type="checkbox" ${voiceSettings.enabled ? 'checked' : ''}> Voice alerts</label>
      <label for="settingsVoiceVolume">Voice volume</label>
      <input id="settingsVoiceVolume" type="range" min="0" max="1" step="0.1" value="${voiceSettings.volume}">
      <label for="settingsVoiceLevel">Voice escalation level</label>
      <select id="settingsVoiceLevel">
        <option value="1">1</option><option value="2">2</option><option value="3">3</option>
      </select>
      <div style="margin-top: 6px; padding-top: 6px; border-top: 1px solid var(--border);">
        <label style="display:block; margin-bottom: 4px; font-size: 0.78rem; color: var(--muted);">Alert Sound Previews</label>
        <div class="sound-test-group">
          <button type="button" class="btn-test-sound" onclick="playAlertSound('closure')">🔊 Eye-Closure Siren</button>
          <button type="button" class="btn-test-sound" onclick="playAlertSound('gaze')">🔔 Head-Turn Chime</button>
        </div>
      </div>
      <div class="settings-actions">
        <button class="btn btn-secondary" onclick="closeSettings()">Cancel</button>
        <button class="btn btn-primary" onclick="saveSettings()">Save</button>
      </div>
    </div>`;
  document.body.appendChild(modal);
  document.getElementById('settingsSensitivity').value = Object.keys(SENSITIVITY).find(key => SENSITIVITY[key].label === sens.label) || '2';
  document.getElementById('settingsVoiceLevel').value = voiceSettings.level;
}

function recordEvaluationSample(ear, faceDetected) {
  evaluation.frames++;
  if (faceDetected) {
    evaluation.faceFrames++;
    evaluation.earTotal += ear;
  }
  updateEvaluation();
}

function updateEvaluation() {
  const hasSamples = evaluation.frames > 0;
  const coverage = hasSamples ? (evaluation.faceFrames / evaluation.frames) * 100 : 0;
  const averageEar = evaluation.faceFrames ? evaluation.earTotal / evaluation.faceFrames : 0;
  const elapsedMinutes = sessionStart ? Math.max((Date.now() - sessionStart) / 60000, 1 / 60) : 0;
  const alertsPerMinute = elapsedMinutes ? alertCount / elapsedMinutes : 0;

  document.getElementById('faceCoverage').textContent = hasSamples ? `${coverage.toFixed(1)}%` : '--';
  document.getElementById('averageEar').textContent = evaluation.faceFrames ? averageEar.toFixed(3) : '--';
  document.getElementById('alertRate').textContent = hasSamples ? alertsPerMinute.toFixed(1) : '--';
  document.getElementById('framesSampled').textContent = evaluation.frames;
  document.getElementById('evaluationScore').textContent = hasSamples ? 'LIVE' : 'READY';

  const status = document.getElementById('evaluationStatus');
  if (!hasSamples) status.textContent = 'Start a session to collect operational telemetry. This is not an accuracy evaluation.';
  else if (coverage >= 90 && currentFps >= 20) status.textContent = 'Stable tracking and frame rate. Validate detection accuracy against labeled data.';
  else if (coverage >= 70) status.textContent = 'Partial tracking coverage. Check lighting and camera position before comparing models.';
  else status.textContent = 'Low tracking coverage. Current session may not be suitable for model comparison.';
}

function resetEvaluation() {
  evaluation = { frames: 0, faceFrames: 0, earTotal: 0 };
  updateEvaluation();
}

function updateClosureMetrics(closed) {
  const now = Date.now();
  if (closed === null) {
    eyeClosedStart = null;
    wasEyeClosed = false;
    document.getElementById('closureDuration').textContent = '—';
    document.getElementById('closureBar').style.width = '0%';
    document.getElementById('closureStatus').textContent = 'Face not tracked';
    return 0;
  }
  if (closed && !wasEyeClosed) eyeClosedStart = now;
  if (!closed && wasEyeClosed && eyeClosedStart) {
    const duration = (now - eyeClosedStart) / 1000;
    if (duration >= 0.5) longClosureCount++;
    document.getElementById('longClosureCount').textContent = longClosureCount;
    eyeClosedStart = null;
  }
  wasEyeClosed = closed;
  const currentDuration = closed && eyeClosedStart ? (now - eyeClosedStart) / 1000 : 0;
  document.getElementById('closureDuration').textContent = `${currentDuration.toFixed(1)}s`;
  document.getElementById('closureBar').style.width = Math.min(100, (currentDuration / 1.5) * 100) + '%';
  document.getElementById('closureStatus').textContent = closed ? 'Eyes closed' : 'Eyes open';
  return currentDuration * 1000;
}

function updateSelectedModel(value) {
  if (isRunning) return;
  selectedModel = value || document.getElementById('modelSelect').value;
  document.getElementById('activeModelValue').textContent = getModelLabel();
}

function getModelLabel() {
  return `MediaPipe v${MEDIAPIPE_VERSION}`;
}

function startDetection() {
  if (isRunning) return;
  selectedModel = document.getElementById('modelSelect').value;
  const runId = ++detectionRunId;
  isRunning = true;
  sessionStart = Date.now();
  fpsFrames = 0;
  fpsLastTime = performance.now();
  lastScoreUpdate = performance.now();
  lastLoggedAt = 0;
  perclosSamples = [];
  earBuffer = [];
  closedFrames = 0;
  blinkTrackingClosed = false;
  blinkTrackingStart = null;
  gazeAwayStart = null;
  gazeAwayAlerted = false;
  gazeAlignmentBuffer = [];
  currentHeadAlignment = 100;
  gazeTurnDirection = 'Frontal';
  if (document.getElementById('gazeAlignmentValue')) document.getElementById('gazeAlignmentValue').textContent = '—';
  if (document.getElementById('gazeAlignmentBar')) {
    document.getElementById('gazeAlignmentBar').style.width = '0%';
    document.getElementById('gazeAlignmentBar').className = 'metric-bar';
  }
  if (document.getElementById('gazeAlignmentStatus')) {
    document.getElementById('gazeAlignmentStatus').textContent = 'Waiting for detection...';
    document.getElementById('gazeAlignmentStatus').style.color = '';
  }
  if (document.getElementById('headAlignmentValue')) document.getElementById('headAlignmentValue').textContent = '—';
  eyeClosedStart = null;
  wasEyeClosed = false;
  currentFps = 0;
  blinkCount = 0;
  longClosureCount = 0;
  alertCount = 0;
  lastAlertTime = 0;
  faceMissingStart = null;
  faceLossAlerted = false;
  drowsinessScore = 0;
  lastInferenceMs = 0;
  evaluation = { frames: 0, faceFrames: 0, earTotal: 0 };
  document.getElementById('blinkCount').textContent = '0';
  document.getElementById('longClosureCount').textContent = '0';
  document.getElementById('alertCount').textContent = '0';
  document.getElementById('sessionTime').textContent = '00:00';
  document.getElementById('fpsValue').textContent = '—';
  document.getElementById('inferenceValue').textContent = '—';
  document.getElementById('faceDetected').textContent = 'No';
  document.getElementById('earValue').textContent = '—';
  document.getElementById('earStatus').textContent = 'Waiting for detection...';
  document.getElementById('closureDuration').textContent = '0.0s';
  document.getElementById('closureBar').style.width = '0%';
  document.getElementById('closureStatus').textContent = 'Eyes open';
  document.getElementById('perclosValue').textContent = '—';
  document.getElementById('perclosBar').style.width = '0%';
  document.getElementById('perclosCoverage').textContent = 'Waiting for face tracking';
  document.getElementById('alertOverlay').classList.add('hidden');
  updateGauge(0);
  document.getElementById('modelSelect').disabled = true;
  document.getElementById('startBtn').classList.add('hidden');
  document.getElementById('stopBtn').classList.remove('hidden');
  document.getElementById('loadingMessage').textContent = 'Loading MediaPipe model...';
  document.getElementById('videoLoading').classList.remove('hidden');
  document.getElementById('activeModelValue').textContent = getModelLabel();
  setStatus('Loading detector...', true);
  updateEvaluation();
  sessionTimer = setInterval(updateSessionTime, 1000);
  startRealDetection(runId);
}

function stopDetection() {
  detectionRunId++;
  isRunning = false;
  if (stream) {
    stream.getTracks().forEach(track => track.stop());
    stream = null;
  }
  const video = document.getElementById('video');
  if (video) video.srcObject = null;
  if (mediaPipeLandmarker) {
    try { mediaPipeLandmarker.close(); } catch (error) { console.warn('Could not close MediaPipe task:', error); }
    mediaPipeLandmarker = null;
  }
  modelsLoaded = false;
  if (animFrameId) cancelAnimationFrame(animFrameId);
  if (sessionTimer) clearInterval(sessionTimer);
  animFrameId = null;
  sessionTimer = null;
  document.getElementById('modelSelect').disabled = false;
  document.getElementById('startBtn').classList.remove('hidden');
  document.getElementById('stopBtn').classList.add('hidden');
  setCameraBadge(false);
  setStatus('Detection stopped', false);
}

function resetStats() {
  stopDetection();
  sessionStart = null;
  blinkCount = 0;
  longClosureCount = 0;
  alertCount = 0;
  drowsinessScore = 0;
  earBuffer = [];
  perclosSamples = [];
  eyeClosedStart = null;
  wasEyeClosed = false;
  lastAlertTime = 0;
  faceMissingStart = null;
  faceLossAlerted = false;
  blinkTrackingClosed = false;
  blinkTrackingStart = null;
  gazeAwayStart = null;
  gazeAwayAlerted = false;
  gazeAlignmentBuffer = [];
  currentHeadAlignment = 100;
  gazeTurnDirection = 'Frontal';
  if (document.getElementById('gazeAlignmentValue')) document.getElementById('gazeAlignmentValue').textContent = '—';
  if (document.getElementById('gazeAlignmentBar')) {
    document.getElementById('gazeAlignmentBar').style.width = '0%';
    document.getElementById('gazeAlignmentBar').className = 'metric-bar';
  }
  if (document.getElementById('gazeAlignmentStatus')) {
    document.getElementById('gazeAlignmentStatus').textContent = 'Waiting for detection...';
    document.getElementById('gazeAlignmentStatus').style.color = '';
  }
  if (document.getElementById('headAlignmentValue')) document.getElementById('headAlignmentValue').textContent = '—';
  updateGauge(0);
  resetEvaluation();
  document.getElementById('blinkCount').textContent = '0';
  document.getElementById('longClosureCount').textContent = '0';
  document.getElementById('alertCount').textContent = '0';
  document.getElementById('sessionTime').textContent = '00:00';
  document.getElementById('fpsValue').textContent = '—';
  document.getElementById('inferenceValue').textContent = '—';
  document.getElementById('faceDetected').textContent = 'No';
  document.getElementById('earValue').textContent = '—';
  document.getElementById('perclosValue').textContent = '—';
  document.getElementById('perclosBar').style.width = '0%';
  document.getElementById('perclosCoverage').textContent = 'Waiting for face tracking';
  document.getElementById('alertOverlay').classList.add('hidden');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

async function loadSelectedModel() {
  const vision = await import(`${MEDIAPIPE_CDN}/vision_bundle.mjs`);
  const fileset = await vision.FilesetResolver.forVisionTasks(`${MEDIAPIPE_CDN}/wasm`);
  mediaPipeLandmarker = await vision.FaceLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: MEDIAPIPE_MODEL_URL },
    runningMode: 'VIDEO',
    numFaces: 1,
    minFaceDetectionConfidence: 0.5,
    minFacePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
    outputFaceBlendshapes: false
  });
}

async function startRealDetection(runId) {
  try {
    await loadSelectedModel();
    if (runId !== detectionRunId) {
      if (mediaPipeLandmarker) mediaPipeLandmarker.close();
      mediaPipeLandmarker = null;
      return;
    }
    modelsLoaded = true;

    if (!navigator.mediaDevices?.getUserMedia) throw new Error('Camera access is unavailable');
    const cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false
    });
    if (runId !== detectionRunId) {
      cameraStream.getTracks().forEach(track => track.stop());
      return;
    }
    stream = cameraStream;
    const video = document.getElementById('video');
    video.srcObject = stream;
    await video.play();
    if (runId !== detectionRunId) return;
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    hideLoading();
    setCameraBadge(true);
    setStatus('Detection active', true);
    detectRealFrame();
  } catch (error) {
    console.error('Face detection could not start:', error);
    if (runId !== detectionRunId) return;
    stopDetection();
    document.getElementById('loadingMessage').textContent = `${error.message}. Check network/camera permissions and try again.`;
    document.getElementById('videoLoading').classList.remove('hidden');
    setStatus('Detection unavailable', false);
  }
}

async function detectRealFrame() {
  if (!isRunning || !modelsLoaded) return;
  const video = document.getElementById('video');
  const inferenceStart = performance.now();
  try {
    let ear = null;
    let landmarks = null;
    let box = null;

    const result = mediaPipeLandmarker.detectForVideo(video, performance.now());
    landmarks = result.faceLandmarks?.[0] || null;
    if (landmarks) {
      const leftEye = [33, 160, 158, 133, 153, 144].map(index => landmarks[index]);
      const rightEye = [362, 385, 387, 263, 373, 380].map(index => landmarks[index]);
      ear = (computeEAR(leftEye) + computeEAR(rightEye)) / 2;
      updateBlinkCount(ear);
      updateGazeAttention(landmarks, ear);
    } else {
      blinkTrackingClosed = false;
      blinkTrackingStart = null;
      gazeAwayStart = null;
      gazeAwayAlerted = false;
      gazeAlignmentBuffer = [];
      if (document.getElementById('gazeAlignmentStatus')) {
        document.getElementById('gazeAlignmentStatus').textContent = 'Face not detected';
        document.getElementById('gazeAlignmentStatus').style.color = '';
      }
      if (document.getElementById('gazeAlignmentValue')) {
        document.getElementById('gazeAlignmentValue').textContent = '—';
      }
      if (document.getElementById('gazeAlignmentBar')) {
        document.getElementById('gazeAlignmentBar').style.width = '0%';
        document.getElementById('gazeAlignmentBar').className = 'metric-bar';
      }
      if (document.getElementById('headAlignmentValue')) {
        document.getElementById('headAlignmentValue').textContent = 'No face';
      }
    }

    lastInferenceMs = performance.now() - inferenceStart;
    document.getElementById('inferenceValue').textContent = `${lastInferenceMs.toFixed(1)} ms`;
    if (ear !== null && Number.isFinite(ear)) {
      drawMediaPipeMesh(landmarks);
      processDetectionSample(ear, true);
    } else {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      processDetectionSample(null, false);
    }
  } catch (error) {
    console.warn('Detection frame failed:', error);
  }
  calcFPS();
  if (isRunning) animFrameId = requestAnimationFrame(detectRealFrame);
}

function processDetectionSample(ear, faceDetected) {
  const now = Date.now();
  let closed = false;
  let perclos = null;

  if (faceDetected) {
    faceMissingStart = null;
    faceLossAlerted = false;
    earBuffer.push(ear);
    if (earBuffer.length > 5) earBuffer.shift();
    const smoothEAR = earBuffer.reduce((sum, value) => sum + value, 0) / earBuffer.length;
    closed = smoothEAR < sens.ear;
    recordEvaluationSample(smoothEAR, true);
    document.getElementById('faceDetected').textContent = 'Yes';
    document.getElementById('earValue').textContent = smoothEAR.toFixed(3);
    document.getElementById('earBar').style.width = Math.min(100, (smoothEAR / 0.45) * 100) + '%';
    document.getElementById('earStatus').textContent = closed ? 'Eyes closing' : 'Eyes open';
    updateSparkline(smoothEAR);
    const closureMs = updateClosureMetrics(closed);
    closedFrames = closed ? closedFrames + 1 : 0;
    if (closed && closureMs >= sens.alertMs) triggerAlert();
    const elapsedSeconds = Math.min((performance.now() - lastScoreUpdate) / 1000, 0.25);
    drowsinessScore = closed
      ? Math.min(100, drowsinessScore + elapsedSeconds * 12)
      : Math.max(0, drowsinessScore - elapsedSeconds * 8);
    lastScoreUpdate = performance.now();
  } else {
    if (faceMissingStart === null) faceMissingStart = now;
    const missingFaceMs = now - faceMissingStart;
    if (missingFaceMs >= NO_FACE_ALERT_MS && !faceLossAlerted) {
      faceLossAlerted = true;
      triggerAlert({
        title: 'FACE NOT DETECTED',
        text: 'You are no longer looking at the camera. Please face the webcam to resume tracking.',
        voiceText: 'Please look at the camera.'
      });
    }
    earBuffer = [];
    closedFrames = 0;
    recordEvaluationSample(0, false);
    updateClosureMetrics(null);
    document.getElementById('faceDetected').textContent = 'No';
    document.getElementById('earValue').textContent = '—';
    document.getElementById('earStatus').textContent = 'No face detected';
    const elapsedSeconds = Math.min((performance.now() - lastScoreUpdate) / 1000, 0.25);
    drowsinessScore = Math.max(0, drowsinessScore - elapsedSeconds * 8);
    lastScoreUpdate = performance.now();
  }

  const perclosResult = updatePerclos(faceDetected, closed, now);
  perclos = perclosResult.value;
  updateGauge(drowsinessScore);
  applyDrowsyState(drowsinessScore);
  if (now - lastLoggedAt >= LOG_INTERVAL_MS) {
    logEvent('sample', {
      model: selectedModel,
      modelVersion: MEDIAPIPE_VERSION,
      perclos60s: perclos === null ? '' : Number(perclos.toFixed(2)),
      trackingCoverage60s: Number(perclosResult.coverage.toFixed(2)),
      inferenceMs: Number(lastInferenceMs.toFixed(2)),
      fps: currentFps,
      earThreshold: sens.ear
    });
    lastLoggedAt = now;
  }
}

function updateBlinkCount(ear) {
  const now = performance.now();
  if (!blinkTrackingClosed && ear < sens.ear) {
    blinkTrackingClosed = true;
    blinkTrackingStart = now;
    return;
  }

  if (blinkTrackingClosed && ear >= sens.ear * 1.15) {
    const duration = now - blinkTrackingStart;
    if (duration >= 40 && duration < 500) {
      blinkCount++;
      document.getElementById('blinkCount').textContent = blinkCount;
    }
    blinkTrackingClosed = false;
    blinkTrackingStart = null;
  }
}

function updateGazeAttention(landmarks, ear) {
  const requiredIndexes = [1, 33, 61, 133, 234, 263, 291, 362, 454];
  if (landmarks.length <= Math.max(...requiredIndexes)) return;

  // ── 1. Head & Facial Feature Visibility Analysis ──────
  // When looking straight: nose, both eyes, lips, and both cheeks/ears are visible and symmetric.
  // When turned left/right: camera sees mostly one side cheek; nose-cheek, eye, and lip spans foreshorten.
  const nose = landmarks[1];
  const leftCheek = landmarks[454];   // User's left cheek outer edge
  const rightCheek = landmarks[234];  // User's right cheek outer edge
  const distNoseToLeftCheek = dist(nose, leftCheek);
  const distNoseToRightCheek = dist(nose, rightCheek);
  const minCheek = Math.min(distNoseToLeftCheek, distNoseToRightCheek);
  const maxCheek = Math.max(distNoseToLeftCheek, distNoseToRightCheek);
  const sCheek = maxCheek > 0 ? (minCheek / maxCheek) : 0;

  // 2. Eye span distances from nose to outer eye corners
  const leftEyeOuter = landmarks[33];
  const rightEyeOuter = landmarks[263];
  const distNoseToLeftEye = dist(nose, leftEyeOuter);
  const distNoseToRightEye = dist(nose, rightEyeOuter);
  const minEye = Math.min(distNoseToLeftEye, distNoseToRightEye);
  const maxEye = Math.max(distNoseToLeftEye, distNoseToRightEye);
  const sEye = maxEye > 0 ? (minEye / maxEye) : 0;

  // 3. Mouth span distances from nose to mouth corners
  const leftMouth = landmarks[61];
  const rightMouth = landmarks[291];
  const distNoseToLeftMouth = dist(nose, leftMouth);
  const distNoseToRightMouth = dist(nose, rightMouth);
  const minMouth = Math.min(distNoseToLeftMouth, distNoseToRightMouth);
  const maxMouth = Math.max(distNoseToLeftMouth, distNoseToRightMouth);
  const sMouth = maxMouth > 0 ? (minMouth / maxMouth) : 0;

  // 4. Eye width foreshortening (apparent width of each eye)
  const leftEyeInner = landmarks[133];
  const rightEyeInner = landmarks[362];
  const widthLeftEye = dist(leftEyeOuter, leftEyeInner);
  const widthRightEye = dist(rightEyeInner, rightEyeOuter);
  const minEyeWidth = Math.min(widthLeftEye, widthRightEye);
  const maxEyeWidth = Math.max(widthLeftEye, widthRightEye);
  const sEyeWidth = maxEyeWidth > 0 ? (minEyeWidth / maxEyeWidth) : 0;

  // 5. Composite facial symmetry (0.0 to 1.0)
  const rawSymmetry = 0.45 * sCheek + 0.25 * sEye + 0.15 * sMouth + 0.15 * sEyeWidth;

  // Scale raw symmetry: 0.85+ maps to 100% frontal face visibility
  let visibilityPct = Math.min(100, Math.round((rawSymmetry / 0.85) * 100));

  // 6. Iris Gaze check (when eyes are open)
  let lookingAwayIris = false;
  const irisIndexes = [468, 469, 470, 471, 472, 473, 474, 475, 476, 477];
  if (ear >= sens.ear && landmarks.length > Math.max(...irisIndexes)) {
    const eyeRegions = [
      { contour: [33, 160, 158, 133, 153, 144], iris: [468, 469, 470, 471, 472] },
      { contour: [362, 385, 387, 263, 373, 380], iris: [473, 474, 475, 476, 477] }
    ];
    const horizontalRatios = [];
    const verticalRatios = [];

    for (const eye of eyeRegions) {
      const contour = eye.contour.map(idx => landmarks[idx]);
      const iris = eye.iris.map(idx => landmarks[idx]);
      const irisX = iris.reduce((sum, pt) => sum + pt.x, 0) / iris.length;
      const irisY = iris.reduce((sum, pt) => sum + pt.y, 0) / iris.length;
      const minX = Math.min(...contour.map(pt => pt.x));
      const maxX = Math.max(...contour.map(pt => pt.x));
      const minY = Math.min(...contour.map(pt => pt.y));
      const maxY = Math.max(...contour.map(pt => pt.y));

      if (maxX > minX && maxY > minY) {
        horizontalRatios.push((irisX - minX) / (maxX - minX));
        verticalRatios.push((irisY - minY) / (maxY - minY));
      }
    }

    if (horizontalRatios.length === 2 && verticalRatios.length === 2) {
      const horizontal = (horizontalRatios[0] + horizontalRatios[1]) / 2;
      const vertical = (verticalRatios[0] + verticalRatios[1]) / 2;
      lookingAwayIris = horizontal < 0.22 || horizontal > 0.78 || vertical < 0.15 || vertical > 0.85;
      if (lookingAwayIris) {
        // Eyes looking away brings overall alignment below 70% threshold
        visibilityPct = Math.min(visibilityPct, 64);
      }
    }
  }

  // 7. Smooth the score across recent frames
  gazeAlignmentBuffer.push(visibilityPct);
  if (gazeAlignmentBuffer.length > 5) gazeAlignmentBuffer.shift();
  const smoothVisibility = Math.round(gazeAlignmentBuffer.reduce((a, b) => a + b, 0) / gazeAlignmentBuffer.length);
  currentHeadAlignment = smoothVisibility;

  // 8. Classify head posture & direction
  if (currentHeadAlignment >= HEAD_ALIGNMENT_THRESHOLD) {
    gazeTurnDirection = 'Frontal (Full Face)';
  } else if (distNoseToLeftCheek < distNoseToRightCheek) {
    gazeTurnDirection = 'Turned Left (Side Cheek Visible)';
  } else if (distNoseToRightCheek < distNoseToLeftCheek) {
    gazeTurnDirection = 'Turned Right (Side Cheek Visible)';
  } else if (lookingAwayIris) {
    gazeTurnDirection = 'Eyes Looking Away';
  } else {
    gazeTurnDirection = 'Turned Away (Side Cheek Visible)';
  }

  // 9. Update UI Metrics
  const gazeValElem = document.getElementById('gazeAlignmentValue');
  const gazeBarElem = document.getElementById('gazeAlignmentBar');
  const gazeStatusElem = document.getElementById('gazeAlignmentStatus');
  const headValElem = document.getElementById('headAlignmentValue');

  if (gazeValElem) gazeValElem.textContent = `${currentHeadAlignment}%`;
  if (gazeBarElem) {
    gazeBarElem.style.width = `${Math.min(100, currentHeadAlignment)}%`;
    gazeBarElem.className = currentHeadAlignment < HEAD_ALIGNMENT_THRESHOLD 
      ? 'metric-bar danger-bar' 
      : 'metric-bar';
  }

  if (headValElem) {
    headValElem.textContent = `${currentHeadAlignment}% · ${gazeTurnDirection.split(' ')[0]}`;
  }

  const isAlertCondition = currentHeadAlignment < HEAD_ALIGNMENT_THRESHOLD;
  const now = Date.now();

  // 10. 70% Restriction Logic: >= 70% is NOT an alert, < 70% IS an alert
  if (!isAlertCondition) {
    if (gazeStatusElem) {
      gazeStatusElem.textContent = `✓ Optimal · Full Face Visible (${currentHeadAlignment}% ≥ 70%)`;
      gazeStatusElem.style.color = 'var(--accent)';
    }
    gazeAwayStart = null;
    gazeAwayAlerted = false;
    return;
  }

  // Below 70% threshold: Alert condition
  if (gazeStatusElem) {
    gazeStatusElem.textContent = `⚠️ Alert · ${gazeTurnDirection} (${currentHeadAlignment}% < 70%)`;
    gazeStatusElem.style.color = 'var(--danger)';
  }

  if (gazeAwayStart === null) gazeAwayStart = now;
  if (!gazeAwayAlerted && now - gazeAwayStart >= GAZE_AWAY_ALERT_MS) {
    gazeAwayAlerted = true;
    triggerAlert({
      title: 'HEAD TURNED / NOT FACING CAMERA',
      text: `Head alignment fell to ${currentHeadAlignment}% (below the 70% threshold). ${gazeTurnDirection}. Please look straight into the camera.`,
      voiceText: 'Please look straight into the camera.',
      soundType: 'gaze'
    });
  }
}

function updatePerclos(faceDetected, closed, now) {
  perclosSamples.push({ time: now, closed: faceDetected ? closed : null });
  const cutoff = now - PERCLOS_WINDOW_MS;
  while (perclosSamples.length > 1 && perclosSamples[1].time <= cutoff) perclosSamples.shift();

  const windowStart = Math.max(cutoff, sessionStart || cutoff);
  const windowDuration = Math.max(1, now - windowStart);
  let observedMs = 0;
  let closedMs = 0;
  for (let index = 0; index < perclosSamples.length; index++) {
    const sample = perclosSamples[index];
    const start = Math.max(sample.time, windowStart);
    const end = Math.min(perclosSamples[index + 1]?.time ?? now, now);
    if (sample.closed === null || end <= start) continue;
    observedMs += end - start;
    if (sample.closed) closedMs += end - start;
  }

  const value = observedMs >= 500 ? (closedMs / observedMs) * 100 : null;
  const coverage = (observedMs / windowDuration) * 100;
  document.getElementById('perclosValue').textContent = value === null ? '—' : `${value.toFixed(1)}%`;
  document.getElementById('perclosBar').style.width = `${Math.min(100, value || 0)}%`;
  document.getElementById('perclosCoverage').textContent = value === null
    ? 'Waiting for sufficient face-tracked time'
    : `Face tracking covered ${Math.min(100, coverage).toFixed(0)}% of this window`;
  return { value, coverage };
}

function updateSessionTime() {
  const elapsed = Math.floor((Date.now() - sessionStart) / 1000);
  const m = String(Math.floor(elapsed / 60)).padStart(2, '0');
  const s = String(elapsed % 60).padStart(2, '0');
  document.getElementById('sessionTime').textContent = `${m}:${s}`;
}

// -------------------------
// Logging and persistence
// -------------------------
function logEvent(type, data = {}) {
  try {
    logs.push(Object.assign({ time: new Date().toISOString(), type }, data));
  } catch (e) { /* ignore */ }
}

function downloadLogs() {
  if (!logs.length) { alert('No logs to export'); return; }
  const keys = [...new Set(logs.flatMap(row => Object.keys(row)))];
  const escapeCsv = value => `"${String(value ?? '').replace(/"/g, '""')}"`;
  const rows = logs.map(row => keys.map(key => escapeCsv(row[key])).join(','));
  const csv = [keys.map(escapeCsv).join(','), ...rows].join('\r\n');
  const blob = new Blob(['\ufeff', csv], { type: 'text/csv;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = 'drowseguard_logs.csv';
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

function clearLogs() {
  if (!confirm('Clear all logs?')) return;
  logs.length = 0;
  alert('Logs cleared');
}

 

function closeSettings() {
  const m = document.getElementById('settingsModal');
  if (m) m.parentElement.remove();
}

function saveSettings() {
  const val = document.getElementById('settingsSensitivity').value;
  const earInput = document.getElementById('settingsEar');
  const earVal = Number(earInput.value);
  sens = { ...SENSITIVITY[val] };
  if (Number.isFinite(earVal) && earVal >= 0.1 && earVal <= 0.5) sens.ear = earVal;
  // voice settings
  const vEnabled = document.getElementById('settingsVoiceEnabled').checked;
  const vVolume = parseFloat(document.getElementById('settingsVoiceVolume').value || 0.8);
  const vLevel  = parseInt(document.getElementById('settingsVoiceLevel').value || 3, 10);
  voiceSettings = { enabled: vEnabled, volume: vVolume, level: vLevel };
  localStorage.setItem('drowseguard_sens', JSON.stringify({ val, ear: sens.ear }));
  document.getElementById('sensitivitySlider').value = val;
  document.getElementById('sensitivityVal').textContent = sens.label;
  document.getElementById('thresholdValue').textContent = sens.ear.toFixed(2);
  localStorage.setItem('drowseguard_voice', JSON.stringify(voiceSettings));
  closeSettings();
}

function loadSettings() {
  try {
    const s = JSON.parse(localStorage.getItem('drowseguard_sens'));
    if (s) {
      const sensitivityKey = SENSITIVITY[s.val] ? s.val : '2';
      sens = { ...SENSITIVITY[sensitivityKey] };
      if (Number.isFinite(Number(s.ear)) && Number(s.ear) >= 0.1 && Number(s.ear) <= 0.5) sens.ear = Number(s.ear);
      document.getElementById('sensitivitySlider').value = sensitivityKey;
      document.getElementById('sensitivityVal').textContent = sens.label;
      document.getElementById('thresholdValue').textContent = sens.ear.toFixed(2);
    }
    const vs = JSON.parse(localStorage.getItem('drowseguard_voice'));
    if (vs) voiceSettings = Object.assign(voiceSettings, vs);
  } catch (e) { /* ignore */ }
}

// -------------------------
// Voice escalation
// -------------------------
function speak(text, opts = {}) {
  if (!window.speechSynthesis) return;
  try {
    const u = new SpeechSynthesisUtterance(text);
    u.volume = opts.volume ?? voiceSettings.volume;
    u.rate = opts.rate || 1.0;
    u.pitch = opts.pitch || 1.0;
    window.speechSynthesis.speak(u);
  } catch (e) { /* ignore */ }
}

function startVoiceEscalation() {
  if (!voiceSettings.enabled) return;
  stopVoiceEscalation();
  currentVoiceLevel = 1;
  speak(voiceMessages[currentVoiceLevel], { volume: voiceSettings.volume });
  voiceTimer = setTimeout(() => escalateVoice(), 4000);
}

function escalateVoice() {
  if (!voiceSettings.enabled) return;
  currentVoiceLevel = Math.min(voiceSettings.level, currentVoiceLevel + 1);
  const msg = voiceMessages[currentVoiceLevel] || voiceMessages[voiceSettings.level];
  // increase urgency via rate/pitch for higher levels
  const opts = { volume: voiceSettings.volume, rate: 1 + (currentVoiceLevel - 1) * 0.12, pitch: 1 + (currentVoiceLevel - 1) * 0.08 };
  speak(msg, opts);
  voiceTimer = setTimeout(() => escalateVoice(), 5000);
}

function stopVoiceEscalation() {
  if (voiceTimer) { clearTimeout(voiceTimer); voiceTimer = null; }
  if (window.speechSynthesis) window.speechSynthesis.cancel();
  currentVoiceLevel = 0;
}

// -------------------------
// EAR sparkline
// -------------------------
function updateSparkline(ear) {
  const c = document.getElementById('earSpark');
  if (!c) return;
  const ctx2 = c.getContext('2d');
  earHistory.push(ear);
  if (earHistory.length > 100) earHistory.shift();
  const w = c.width, h = c.height;
  ctx2.clearRect(0,0,w,h);
  // background
  ctx2.fillStyle = 'rgba(255,255,255,0.02)'; ctx2.fillRect(0,0,w,h);
  // path
  ctx2.beginPath();
  for (let i=0;i<earHistory.length;i++) {
    const x = (i/(earHistory.length-1 || 1))*w;
    const v = earHistory[i];
    const y = h - (v/0.45)*h;
    if (i===0) ctx2.moveTo(x,y); else ctx2.lineTo(x,y);
  }
  ctx2.strokeStyle = 'rgba(0,212,170,0.9)'; ctx2.lineWidth = 2; ctx2.stroke();
}

// Eye Aspect Ratio calculation
function computeEAR(eye) {
  const A = dist(eye[1], eye[5]);
  const B = dist(eye[2], eye[4]);
  const C = dist(eye[0], eye[3]);
  return C > 0 ? (A + B) / (2.0 * C) : 0;
}
function dist(p1, p2) {
  return Math.hypot(p1.x - p2.x, p1.y - p2.y);
}

// ── Draw Face Mesh ────────────────────────────────────
function drawFaceMesh(landmarks, box) {
  const pts = landmarks.positions;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Face box glow
  const isDrowsy = drowsinessScore > 50;
  ctx.strokeStyle = isDrowsy ? 'rgba(239,68,68,0.6)' : 'rgba(0,212,170,0.5)';
  ctx.lineWidth = 2;
  ctx.roundRect(box.x, box.y, box.width, box.height, 8);
  ctx.stroke();

  // Draw landmark dots
  ctx.fillStyle = isDrowsy ? 'rgba(239,68,68,0.8)' : 'rgba(0,212,170,0.8)';
  pts.forEach(p => {
    ctx.beginPath();
    ctx.arc(p.x, p.y, 1.5, 0, 2 * Math.PI);
    ctx.fill();
  });

  // Draw eye contours
  drawContour(landmarks.getLeftEye(),  isDrowsy);
  drawContour(landmarks.getRightEye(), isDrowsy);
}

function drawMediaPipeMesh(landmarks) {
  const points = landmarks.map(point => ({ x: point.x * canvas.width, y: point.y * canvas.height }));
  const xs = points.map(point => point.x);
  const ys = points.map(point => point.y);
  const isDrowsy = drowsinessScore > 50;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = isDrowsy ? 'rgba(239,68,68,0.6)' : 'rgba(0,212,170,0.5)';
  ctx.lineWidth = 2;
  ctx.roundRect(Math.min(...xs), Math.min(...ys), Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys), 8);
  ctx.stroke();
  ctx.fillStyle = isDrowsy ? 'rgba(239,68,68,0.75)' : 'rgba(0,212,170,0.75)';
  points.forEach((point, index) => {
    if (index % 6 !== 0) return;
    ctx.beginPath();
    ctx.arc(point.x, point.y, 1.2, 0, 2 * Math.PI);
    ctx.fill();
  });
  drawContour([33, 160, 158, 133, 153, 144].map(index => points[index]), isDrowsy);
  drawContour([362, 385, 387, 263, 373, 380].map(index => points[index]), isDrowsy);

  // Draw Head Alignment HUD Badge on camera overlay
  drawHeadAlignmentHUD();
}

function drawHeadAlignmentHUD() {
  ctx.save();
  // Un-mirror horizontally because overlay-canvas is mirrored with transform: scaleX(-1) in CSS
  ctx.translate(canvas.width, 0);
  ctx.scale(-1, 1);

  const isAlignmentOk = currentHeadAlignment >= HEAD_ALIGNMENT_THRESHOLD;
  const badgeText = isAlignmentOk 
    ? `Head Alignment: ${currentHeadAlignment}% (≥70%) · Frontal` 
    : `⚠️ Head Alignment: ${currentHeadAlignment}% (<70%) · ${gazeTurnDirection}`;

  ctx.font = '600 12px Inter, sans-serif';
  const textWidth = ctx.measureText(badgeText).width;
  const badgeW = textWidth + 26;
  const badgeH = 26;
  const badgeX = 14;
  const badgeY = 14;

  ctx.fillStyle = isAlignmentOk ? 'rgba(6, 13, 24, 0.85)' : 'rgba(239, 68, 68, 0.9)';
  ctx.beginPath();
  if (typeof ctx.roundRect === 'function') {
    ctx.roundRect(badgeX, badgeY, badgeW, badgeH, 8);
  } else {
    ctx.rect(badgeX, badgeY, badgeW, badgeH);
  }
  ctx.fill();
  ctx.strokeStyle = isAlignmentOk ? 'rgba(0, 212, 170, 0.5)' : 'rgba(255, 255, 255, 0.7)';
  ctx.lineWidth = 1;
  ctx.stroke();

  // Status dot
  ctx.fillStyle = isAlignmentOk ? '#00d4aa' : '#ffffff';
  ctx.beginPath();
  ctx.arc(badgeX + 12, badgeY + 13, 4, 0, 2 * Math.PI);
  ctx.fill();

  // Text
  ctx.fillStyle = '#ffffff';
  ctx.fillText(badgeText, badgeX + 22, badgeY + 17);
  ctx.restore();
}

function drawContour(points, danger) {
  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);
  points.forEach(p => ctx.lineTo(p.x, p.y));
  ctx.closePath();
  ctx.strokeStyle = danger ? 'rgba(239,68,68,0.9)' : 'rgba(0,212,170,0.9)';
  ctx.lineWidth = 1.5;
  ctx.stroke();
  ctx.fillStyle = danger ? 'rgba(239,68,68,0.08)' : 'rgba(0,212,170,0.08)';
  ctx.fill();
}

// ── Simulated Demo Mode ───────────────────────────────
function runSimulatedLoop() {
  let t = 0;
  function step() {
    if (!isRunning) return;
    t += 0.02;
    // Simulate a sin-wave EAR that occasionally dips
    const ear = 0.30 + 0.12 * Math.sin(t) - (Math.sin(t * 0.3) > 0.85 ? 0.18 : 0);
    earBuffer.push(ear);
    if (earBuffer.length > 8) earBuffer.shift();
    const smoothEAR = earBuffer.reduce((a,b) => a+b) / earBuffer.length;

    document.getElementById('earValue').textContent = smoothEAR.toFixed(3);
    const earPct = Math.min(100, (smoothEAR / 0.45) * 100);
    document.getElementById('earBar').style.width = earPct + '%';
    document.getElementById('faceDetected').textContent = 'Sim ✓';
    recordEvaluationSample(smoothEAR, true);

    // Simulated Head Alignment (mostly 88-96% frontal, occasionally dips)
    const simAlignment = Math.round(92 + 6 * Math.cos(t * 0.5));
    currentHeadAlignment = simAlignment;
    if (document.getElementById('gazeAlignmentValue')) document.getElementById('gazeAlignmentValue').textContent = `${simAlignment}%`;
    if (document.getElementById('gazeAlignmentBar')) {
      document.getElementById('gazeAlignmentBar').style.width = `${simAlignment}%`;
      document.getElementById('gazeAlignmentBar').className = 'metric-bar';
    }
    if (document.getElementById('gazeAlignmentStatus')) {
      document.getElementById('gazeAlignmentStatus').textContent = `✓ Optimal · Full Face Visible (${simAlignment}% ≥ 70%)`;
      document.getElementById('gazeAlignmentStatus').style.color = 'var(--accent)';
    }
    if (document.getElementById('headAlignmentValue')) {
      document.getElementById('headAlignmentValue').textContent = `Frontal (${simAlignment}%)`;
    }

    const closed = smoothEAR < sens.ear;
    if (closed) {
      drowsinessScore = Math.min(100, drowsinessScore + 2);
      closedFrames++;
      document.getElementById('earStatus').textContent = '⚠️ Eyes closing (demo)';
    } else {
      drowsinessScore = Math.max(0, drowsinessScore - 1);
      closedFrames = 0;
      document.getElementById('earStatus').textContent = '✓ Eyes open (demo)';
    }
    if (closedFrames >= sens.frames) triggerAlert({ soundType: 'closure' });

    updateGauge(drowsinessScore);
    applyDrowsyState(drowsinessScore);
    calcFPS();
    animFrameId = requestAnimationFrame(step);
  }
  step();
}

// ── UI Helpers ────────────────────────────────────────
function updateGauge(score) {
  const arc   = document.getElementById('gaugeArc');
  const total = 251;
  const offset = total - (score / 100) * total;
  arc.style.strokeDashoffset = offset;

  document.getElementById('gaugePercent').textContent = Math.round(score);
  const lbl = document.getElementById('gaugeLabel');
  if (score < 30)      { lbl.textContent = 'Alert';    lbl.style.color = '#00d4aa'; }
  else if (score < 60) { lbl.textContent = 'Drowsy';   lbl.style.color = '#f59e0b'; }
  else                 { lbl.textContent = 'Danger!';  lbl.style.color = '#ef4444'; }
}

function applyDrowsyState(score) {
  document.body.classList.remove('drowsy-warn', 'drowsy-danger');
  if (score >= 60) document.body.classList.add('drowsy-danger');
  else if (score >= 30) document.body.classList.add('drowsy-warn');

  const dot = document.getElementById('statusDot');
  if (score >= 60)      { dot.className = 'pulse-dot danger'; }
  else if (score >= 30) { dot.className = 'pulse-dot'; dot.style.background = '#f59e0b'; }
  else if (isRunning)   { dot.className = 'pulse-dot active'; }
}

function decayDrowsiness() {
  drowsinessScore = Math.max(0, drowsinessScore - 0.5);
  updateGauge(drowsinessScore);
}

function triggerAlert({ title = 'PROLONGED EYE CLOSURE', text = 'The configured eye-closure rule was reached. This research prototype cannot diagnose drowsiness.', voiceText = null, soundType = 'closure' } = {}) {
  const now = Date.now();
  if (now - lastAlertTime < 8000) return;   // throttle alerts
  lastAlertTime = now;
  alertCount++;
  document.getElementById('alertCount').textContent = alertCount;

  document.getElementById('alertTitle').textContent = title;
  document.getElementById('alertText').textContent = text;

  logEvent('alert', {
    model: selectedModel,
    modelVersion: MEDIAPIPE_VERSION,
    score: Number(drowsinessScore.toFixed(1)),
    closedFrames,
    closureDurationMs: eyeClosedStart ? Date.now() - eyeClosedStart : 0,
    perclos60s: document.getElementById('perclosValue').textContent,
    earThreshold: sens.ear,
    alertType: title,
    headAlignment: currentHeadAlignment
  });

  playAlertSound(soundType);
  document.getElementById('alertOverlay').classList.remove('hidden');
  closedFrames = 0;
  if (voiceText) {
    speak(voiceText, { volume: voiceSettings.volume });
  } else {
    startVoiceEscalation();
  }
}

function dismissAlert() {
  document.getElementById('alertOverlay').classList.add('hidden');
  drowsinessScore = Math.max(0, drowsinessScore - 30);
  stopVoiceEscalation();
}

function playAlertSound(type = 'closure') {
  try {
    const ctx2 = new (window.AudioContext || window.webkitAudioContext)();
    const now = ctx2.currentTime;

    if (type === 'gaze') {
      // 🔔 Distinct Dual-Tone Attention Chime for Head Turn / Gaze (Lane assist / distraction chime)
      // Tone 1: Crisp high tone
      const osc1 = ctx2.createOscillator();
      const gain1 = ctx2.createGain();
      osc1.connect(gain1);
      gain1.connect(ctx2.destination);
      osc1.type = 'triangle';
      osc1.frequency.setValueAtTime(880, now); // A5
      osc1.frequency.exponentialRampToValueAtTime(659.25, now + 0.18); // E5
      gain1.gain.setValueAtTime(0.35, now);
      gain1.gain.linearRampToValueAtTime(0.01, now + 0.22);
      osc1.start(now);
      osc1.stop(now + 0.22);

      // Tone 2: Higher bright chime immediately following
      const osc2 = ctx2.createOscillator();
      const gain2 = ctx2.createGain();
      osc2.connect(gain2);
      gain2.connect(ctx2.destination);
      osc2.type = 'triangle';
      osc2.frequency.setValueAtTime(1174.66, now + 0.16); // D6
      osc2.frequency.exponentialRampToValueAtTime(880, now + 0.45);
      gain2.gain.setValueAtTime(0.35, now + 0.16);
      gain2.gain.linearRampToValueAtTime(0.01, now + 0.50);
      osc2.start(now + 0.16);
      osc2.stop(now + 0.50);
    } else {
      // 🚨 Urgent Rising Siren for Eye Closure / Drowsiness
      const osc  = ctx2.createOscillator();
      const gain = ctx2.createGain();
      osc.connect(gain);
      gain.connect(ctx2.destination);
      osc.type = 'sawtooth';
      osc.frequency.setValueAtTime(440, now);
      osc.frequency.linearRampToValueAtTime(880, now + 0.3);
      gain.gain.setValueAtTime(0.4, now);
      gain.gain.linearRampToValueAtTime(0, now + 0.8);
      osc.start(now);
      osc.stop(now + 0.8);
    }
  } catch(e) {
    console.warn('AudioContext alert failed:', e);
  }
}
window.playAlertSound = playAlertSound;

function updateStatCards(ear, closure, earSt, closeSt) {
  if (earSt)   document.getElementById('earStatus').textContent    = earSt;
  if (closeSt) document.getElementById('closureStatus').textContent = closeSt;
}

function setStatus(msg, active) {
  document.getElementById('statusText').textContent = msg;
  const dot = document.getElementById('statusDot');
  dot.className = active ? 'pulse-dot active' : 'pulse-dot';
  if (!active) dot.style.background = '';
}

function setCameraBadge(active) {
  const badge = document.getElementById('cameraBadge');
  badge.classList.toggle('live', active);
  document.getElementById('cameraBadgeText').textContent = active ? 'CAMERA ACTIVE' : 'CAMERA OFF';
}

function hideLoading() {
  document.getElementById('videoLoading').classList.add('hidden');
}


function calcFPS() {
  fpsFrames++;
  const now = performance.now();
  if (now - fpsLastTime >= 1000) {
    currentFps = fpsFrames;
    fpsFrames  = 0;
    fpsLastTime = now;
    document.getElementById('fpsValue').textContent = currentFps + ' fps';
    updateEvaluation();
  }
}

function updateSensitivity(val) {
  sens = { ...SENSITIVITY[val] };
  const labels = { 1: 'Low', 2: 'Medium', 3: 'High' };
  document.getElementById('sensitivityVal').textContent = labels[val];
  document.getElementById('thresholdValue').textContent = sens.ear.toFixed(2);
}

function initializeApp() {
  loadSettings();
  updateSelectedModel(document.getElementById('modelSelect').value);
  setCameraBadge(false);
  hideLoading();
  setStatus('Ready', false);
  updateEvaluation();
}

initializeApp();
