/* =====================================================
   DrowseGuard – Driver Drowsiness Detection Engine
   ===================================================== */

// ── Constants ─────────────────────────────────────────
const MEDIAPIPE_VERSION   = '1.1.0';
const MEDIAPIPE_CDN       = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}`;
const MEDIAPIPE_MODEL_URL  = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';
const ONNX_RUNTIME_VERSION = '1.20.1';
const ONNX_RUNTIME_CDN     = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ONNX_RUNTIME_VERSION}/dist/`;
const YOLO_INPUT_SIZE      = 640;
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
let videoFrameCallbackId = null;
let lastVideoTime    = -1;
let lastSparklineUpdate = 0;
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
const MAR_THRESHOLD            = 0.60; // Mouth Aspect Ratio threshold for yawning alert
let yawnEpisodeActive = false;
let yawnAlerted = false;
let isProcessingFrame = false;        // Concurrency guard to eliminate video lag completely
let marBuffer        = [];            // Rolling buffer for MAR smoothing
let currentMAR       = 0.20;          // Current computed mouth aspect ratio
let yawnFrames       = 0;             // Consecutive frames with open mouth (yawn)
let yawnCount        = 0;
let fpsFrames        = 0;
let fpsLastTime      = performance.now();
let currentFps       = 0;
let sens             = SENSITIVITY[2];
let modelsLoaded     = false;
let mediaPipeLandmarker = null;
let selectedModel    = 'mediapipe';
let detectionRunId   = 0;
let onnxRuntimePromise = null;
let yoloModels = {
  yolov8: { file: null, session: null },
  yolo11: { file: null, session: null }
};
let yoloDetections = [];
let yoloInferenceMs = 0;
let yoloTargetAlerted = false;
let yoloInputCanvas = null;
let yoloInputContext = null;
let perclosSamples   = [];
let lastLoggedAt     = 0;
let lastInferenceMs  = 0;
let lastScoreUpdate  = 0;
let evaluation = { frames: 0, faceFrames: 0, earTotal: 0 };
let logs             = [];
let earHistory       = [];
let voiceSettings    = { enabled: true, volume: 0.8 };
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
      <div class="settings-actions">
        <button class="btn btn-secondary" onclick="closeSettings()">Cancel</button>
        <button class="btn btn-primary" onclick="saveSettings()">Save</button>
      </div>
    </div>`;
  document.body.appendChild(modal);
  document.getElementById('settingsSensitivity').value = Object.keys(SENSITIVITY).find(key => SENSITIVITY[key].label === sens.label) || '2';
}

function recordEvaluationSample(ear, faceDetected) {
  evaluation.frames++;
  if (faceDetected) {
    evaluation.faceFrames++;
    evaluation.earTotal += ear;
  }
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
  if (selectedModel === 'yolo8') selectedModel = 'yolov8';
  const task = selectedModel === 'yolov8' || selectedModel === 'yolo11'
    ? document.getElementById(`${selectedModel}Task`).value
    : 'detection';
  const isClassifier = task === 'classification';

  document.body.classList.remove('yolo-theme', 'theme-yolo8', 'theme-yolo11');
  const badgeTag = document.getElementById('modelBadgeTag');
  const modelNote = document.getElementById('modelNote');
  const yoloSettings = document.getElementById('yoloModelSettings');
  const yolo8Config = document.getElementById('yolov8Config');
  const yolo11Config = document.getElementById('yolo11Config');
  const isYoloMode = selectedModel === 'yolov8' || selectedModel === 'yolo11';
  document.body.classList.toggle('yolo-only-mode', isYoloMode);
  document.body.classList.toggle('yolo-classifier-mode', isClassifier);
  if (yoloSettings) yoloSettings.classList.toggle('hidden', !isYoloMode);
  if (yolo8Config) yolo8Config.classList.toggle('hidden', selectedModel !== 'yolov8');
  if (yolo11Config) yolo11Config.classList.toggle('hidden', selectedModel !== 'yolo11');

  if (selectedModel === 'yolov8' || selectedModel === 'yolo8') {
    selectedModel = 'yolov8';
    document.body.classList.add('yolo-theme', 'theme-yolo8');
    const loaded = Boolean(yoloModels.yolov8.file);
    if (badgeTag) { badgeTag.textContent = loaded ? 'ONNX selected' : 'Add ONNX model'; badgeTag.style.color = '#fb923c'; }
      if (modelNote) modelNote.textContent = loaded
      ? isClassifier
        ? 'YOLOv8 classifies the full frame. Eye-state estimates are experimental; gaze is unavailable.'
        : 'YOLOv8 detects its configured classes. Eye, blink, and gaze metrics are unavailable in this mode.'
      : isClassifier
        ? 'Select an experimental YOLOv8 eye-state classifier ONNX file below.'
        : 'Select a compatible YOLOv8 ONNX model below.';
  } else if (selectedModel === 'yolo11') {
    document.body.classList.add('yolo-theme', 'theme-yolo11');
    const loaded = Boolean(yoloModels.yolo11.file);
    if (badgeTag) { badgeTag.textContent = loaded ? 'ONNX selected' : 'Add ONNX model'; badgeTag.style.color = '#00f2fe'; }
    if (modelNote) modelNote.textContent = loaded
    ? isClassifier
      ? 'YOLO11 classifies the full frame. Eye-state estimates are experimental; gaze is unavailable.'
      : 'YOLO11 detects its configured classes. Eye, blink, and gaze metrics are unavailable in this mode.'
    : isClassifier
      ? 'Select an experimental YOLO11 eye-state classifier ONNX file below.'
      : 'Select a compatible YOLO11 ONNX model below.';
  } else {
    selectedModel = 'mediapipe';
    if (badgeTag) { badgeTag.textContent = 'Baseline'; badgeTag.style.color = '#60a5fa'; }
    if (modelNote) modelNote.textContent = 'MediaPipe / Face API Active: Standard facial landmark baseline with original split-panel theme.';
  }

  ['mediapipe', 'yolov8', 'yolo11'].forEach(m => {
    const row = document.getElementById(`benchRow-${m}`);
    if (row) {
      row.classList.toggle('bench-active', m === selectedModel);
    }
  });
  const mediapipeStatus = document.getElementById('benchStatus-mediapipe');
  if (mediapipeStatus) {
    mediapipeStatus.textContent = selectedModel === 'mediapipe'
      ? 'Active · EAR · MAR · landmark alignment'
      : 'Not loaded in YOLO-only mode';
  }
  const prototypeNotice = document.getElementById('prototypeNotice');
  if (prototypeNotice) {
    prototypeNotice.textContent = isYoloMode
      ? isClassifier
        ? 'Experimental full-frame classifier. It does not locate eyes, detect gaze-away, or validate real-world drowsiness.'
        : 'YOLO detects only its model classes. Use MediaPipe mode for eye, blink, and gaze alerts.'
      : 'MediaPipe provides eye, blink, mouth, and gaze measurements.';
  }
  document.querySelectorAll('.mp-only').forEach(element => {
    const showClassifierMetric = isClassifier && element.classList.contains('classifier-metric');
    element.classList.toggle('hidden', isYoloMode && !showClassifierMetric);
  });
  const perclosLabel = document.getElementById('perclosMetricLabel');
  if (perclosLabel) perclosLabel.textContent = isClassifier
    ? 'PERCLOS · experimental classifier estimate'
    : 'PERCLOS · rolling 60 s';
  const earLabel = document.getElementById('earMetricLabel');
  if (earLabel) earLabel.textContent = isClassifier ? 'Top eye-state confidence' : 'Eye Aspect Ratio (EAR)';
  const blinkLabel = document.getElementById('blinkMetricLabel');
  if (blinkLabel) blinkLabel.textContent = isClassifier ? 'Transitions (approx.)' : 'Blinks';
  const scoreMetricLabel = document.getElementById('yoloScoreMetricLabel');
  if (scoreMetricLabel) scoreMetricLabel.textContent = isClassifier
    ? 'Eye-state class score'
    : 'YOLO detection confidence';
  if (isClassifier) {
    document.getElementById('perclosCoverage').textContent = 'Waiting for classifier frames';
    document.getElementById('earStatus').textContent = 'Waiting for classifier output';
  } else {
    document.getElementById('perclosCoverage').textContent = 'Waiting for face tracking';
    document.getElementById('earStatus').textContent = 'Waiting for detection...';
  }

  const activeModelVal = document.getElementById('activeModelValue');
  if (activeModelVal) activeModelVal.textContent = getModelLabel();
  const inferLabel = document.getElementById('inferenceMetricLabel');
  if (inferLabel) inferLabel.textContent = isYoloMode
    ? isClassifier ? 'YOLO classifier inference time' : 'YOLO inference time'
    : 'MediaPipe inference time';
  const detectionsLabel = document.getElementById('faceDetectedLabel');
  if (detectionsLabel) detectionsLabel.textContent = isYoloMode
    ? isClassifier ? 'Frame classification' : 'YOLO detections'
    : 'Face Detected';
  const headAlignment = document.getElementById('headAlignmentValue');
  if (headAlignment) headAlignment.textContent = isYoloMode ? 'Unavailable' : '—';
  const gazeValue = document.getElementById('gazeAlignmentValue');
  const gazeStatus = document.getElementById('gazeAlignmentStatus');
  const gazeBar = document.getElementById('gazeAlignmentBar');
  if (gazeValue) gazeValue.textContent = isYoloMode ? 'Unavailable' : '—';
  if (gazeStatus) {
    gazeStatus.textContent = isYoloMode
      ? 'Gaze percentage is not available in YOLO mode'
      : 'Waiting for detection...';
    gazeStatus.style.color = '';
  }
  if (gazeBar) {
    gazeBar.style.width = '0%';
    gazeBar.className = 'metric-bar';
  }
  const coverageLabel = document.getElementById('coverageMetricLabel');
  if (coverageLabel) coverageLabel.textContent = isYoloMode
    ? isClassifier ? 'Classifier frame coverage' : 'Detection coverage'
    : 'Face coverage';
  const averageLabel = document.getElementById('averageMetricLabel');
  if (averageLabel) averageLabel.textContent = isYoloMode
    ? isClassifier ? 'Top-class score' : 'Target confidence'
    : 'Avg. EAR';
  const faceDetected = document.getElementById('faceDetected');
  if (faceDetected) faceDetected.textContent = isYoloMode ? '—' : 'No';
  const confidenceEl = document.getElementById('yoloConfValue');
  if (confidenceEl) confidenceEl.textContent = isYoloMode ? '—' : 'N/A';
}

function getModelLabel() {
  if (selectedModel === 'yolo11') {
    return document.getElementById('yolo11Task').value === 'classification'
      ? 'YOLO11 eye-state classifier'
      : 'YOLO11 ONNX detector';
  }
  if (selectedModel === 'yolov8') {
    return document.getElementById('yolov8Task').value === 'classification'
      ? 'YOLOv8 eye-state classifier'
      : 'YOLOv8 ONNX detector';
  }
  return `MediaPipe v${MEDIAPIPE_VERSION}`;
}

async function handleYoloModelFile(model, input) {
  const file = input.files?.[0];
  if (!file) return;
  const status = document.getElementById(`${model}FileStatus`);
  if (!file.name.toLowerCase().endsWith('.onnx')) {
    status.textContent = 'Select an .onnx model file.';
    status.dataset.state = 'error';
    input.value = '';
    return;
  }

  const modelState = yoloModels[model];
  if (modelState.session) {
    try {
      await modelState.session.release();
    } catch (error) {
      console.error(`Could not release the previous ${model} session:`, error);
      status.textContent = 'Could not release the previous model. Reload the page and try again.';
      status.dataset.state = 'error';
      input.value = '';
      return;
    }
  }
  modelState.file = file;
  modelState.session = null;
  yoloDetections = [];
  status.textContent = `${file.name} selected (${(file.size / (1024 * 1024)).toFixed(1)} MB); ready to load`;
  status.dataset.state = 'ready';
  const inferenceCell = document.getElementById(`benchInf-${model}`);
  const benchmarkStatus = document.getElementById(`benchStatus-${model}`);
  if (inferenceCell) inferenceCell.textContent = 'Ready to load';
  if (benchmarkStatus) benchmarkStatus.textContent = 'Waiting to start';
  if (selectedModel === model) updateSelectedModel(model);
}

function startDetection() {
  if (isRunning) return;
  selectedModel = document.getElementById('modelSelect').value;
  if ((selectedModel === 'yolov8' || selectedModel === 'yolo11') && !yoloModels[selectedModel].file) {
    const status = document.getElementById(`${selectedModel}FileStatus`);
    status.textContent = `Choose a ${selectedModel === 'yolov8' ? 'YOLOv8' : 'YOLO11'} ONNX file before starting detection.`;
    status.dataset.state = 'error';
    setStatus('YOLO ONNX model required', false);
    document.getElementById(`${selectedModel}File`).focus();
    return;
  }
  if (selectedModel === 'yolov8' || selectedModel === 'yolo11') {
    try {
      getYoloConfig(selectedModel);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      document.getElementById('modelNote').textContent = message;
      setStatus('Invalid YOLO settings', false);
      return;
    }
  }
  const runId = ++detectionRunId;
  isRunning = true;
  sessionStart = Date.now();
  fpsFrames = 0;
  fpsLastTime = performance.now();
  lastScoreUpdate = performance.now();
  lastLoggedAt = 0;
  lastVideoTime = -1;
  perclosSamples = [];
  earBuffer = [];
  closedFrames = 0;
  blinkTrackingClosed = false;
  blinkTrackingStart = null;
  gazeAwayStart = null;
  gazeAwayAlerted = false;
  yoloDetections = [];
  yoloInferenceMs = 0;
  yoloTargetAlerted = false;
  marBuffer = [];
  currentMAR = 0.20;
  yawnFrames = 0;
  yawnCount = 0;
  yawnEpisodeActive = false;
  yawnAlerted = false;
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
  document.getElementById('yawnCount').textContent = '0';
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
  document.getElementById('benchInf-mediapipe').textContent = '—';
  document.getElementById('benchFps-mediapipe').textContent = '—';
  document.getElementById('faceDetected').textContent = selectedModel === 'mediapipe' ? 'No' : '—';
  document.getElementById('earValue').textContent = '—';
  document.getElementById('earStatus').textContent = 'Waiting for detection...';
  document.getElementById('marValue').textContent = '—';
  document.getElementById('marBar').style.width = '0%';
  document.getElementById('marStatus').textContent = 'Waiting for MediaPipe landmarks';
  document.getElementById('yoloConfValue').textContent = selectedModel === 'mediapipe' ? 'N/A' : '—';
  ['yolov8', 'yolo11'].forEach(model => {
    const inference = document.getElementById(`benchInf-${model}`);
    const fps = document.getElementById(`benchFps-${model}`);
    const status = document.getElementById(`benchStatus-${model}`);
    if (inference) inference.textContent = yoloModels[model].session ? 'Ready' : 'No model loaded';
    if (fps) fps.textContent = '—';
    if (status) status.textContent = yoloModels[model].session ? 'Loaded; waiting for selection' : 'Waiting for ONNX model';
  });
  document.getElementById('closureDuration').textContent = '0.0s';
  document.getElementById('closureBar').style.width = '0%';
  document.getElementById('closureStatus').textContent = 'Eyes open';
  document.getElementById('perclosValue').textContent = '—';
  document.getElementById('perclosBar').style.width = '0%';
  document.getElementById('perclosCoverage').textContent = 'Waiting for face tracking';
  document.getElementById('alertOverlay').classList.add('hidden');
  updateGauge(0);
  document.getElementById('modelSelect').disabled = true;
  document.querySelectorAll('#yoloModelSettings input, #yoloModelSettings select').forEach(input => { input.disabled = true; });
  document.getElementById('startBtn').classList.add('hidden');
  document.getElementById('stopBtn').classList.remove('hidden');
  document.getElementById('loadingMessage').textContent =
    selectedModel === 'mediapipe' ? 'Loading MediaPipe model...' : `Loading ${selectedModel} ONNX model...`;
  document.getElementById('videoLoading').classList.remove('hidden');
  document.getElementById('activeModelValue').textContent = getModelLabel();
  setStatus(selectedModel === 'mediapipe' ? 'Loading MediaPipe detector...' : `Loading ${selectedModel} detector...`, true);
  updateEvaluation();
  sessionTimer = setInterval(updateSessionTime, 1000);
  startRealDetection(runId);
}

function stopDetection() {
  detectionRunId++;
  isRunning = false;
  const video = document.getElementById('video');
  if (videoFrameCallbackId !== null && typeof video?.cancelVideoFrameCallback === 'function') {
    video.cancelVideoFrameCallback(videoFrameCallbackId);
  }
  videoFrameCallbackId = null;
  if (stream) {
    stream.getTracks().forEach(track => track.stop());
    stream = null;
  }
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
  document.querySelectorAll('#yoloModelSettings input, #yoloModelSettings select').forEach(input => { input.disabled = false; });
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
  currentFps = 0;
  lastInferenceMs = 0;
  yoloDetections = [];
  yoloInferenceMs = 0;
  yoloTargetAlerted = false;
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
  marBuffer = [];
  currentMAR = 0.20;
  yawnFrames = 0;
  yawnCount = 0;
  yawnEpisodeActive = false;
  yawnAlerted = false;
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
  document.getElementById('benchInf-mediapipe').textContent = '—';
  document.getElementById('benchFps-mediapipe').textContent = '—';
  document.getElementById('faceDetected').textContent = selectedModel === 'mediapipe' ? 'No' : '—';
  document.getElementById('earValue').textContent = '—';
  document.getElementById('marValue').textContent = '—';
  document.getElementById('marBar').style.width = '0%';
  document.getElementById('marStatus').textContent = 'Waiting for MediaPipe landmarks';
  document.getElementById('yawnCount').textContent = '0';
  document.getElementById('yoloConfValue').textContent = selectedModel === 'mediapipe' ? 'N/A' : '—';
  ['yolov8', 'yolo11'].forEach(model => {
    const inference = document.getElementById(`benchInf-${model}`);
    const fps = document.getElementById(`benchFps-${model}`);
    const status = document.getElementById(`benchStatus-${model}`);
    if (inference) inference.textContent = yoloModels[model].session ? 'Ready' : 'No model loaded';
    if (fps) fps.textContent = '—';
    if (status) status.textContent = yoloModels[model].session ? 'Loaded; waiting for selection' : 'Waiting for ONNX model';
  });
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

function loadOnnxRuntime() {
  if (window.ort) return Promise.resolve(window.ort);
  if (onnxRuntimePromise) return onnxRuntimePromise;

  onnxRuntimePromise = new Promise((resolve, reject) => {
    const script = document.createElement('script');
    script.src = `${ONNX_RUNTIME_CDN}ort.min.js`;
    script.async = true;
    script.onload = () => {
      if (!window.ort) {
        onnxRuntimePromise = null;
        reject(new Error('ONNX Runtime loaded without its browser API.'));
        return;
      }
      window.ort.env.wasm.wasmPaths = ONNX_RUNTIME_CDN;
      window.ort.env.wasm.numThreads = 1;
      resolve(window.ort);
    };
    script.onerror = () => {
      onnxRuntimePromise = null;
      reject(new Error('Could not load ONNX Runtime Web. Check your internet connection and retry.'));
    };
    document.head.appendChild(script);
  });
  return onnxRuntimePromise;
}

async function loadYoloModel(model) {
  const modelState = yoloModels[model];
  if (modelState.session) return modelState.session;
  if (!modelState.file) throw new Error(`Select a ${model} ONNX weights file before starting detection.`);

  const status = document.getElementById(`${model}FileStatus`);
  status.textContent = `Loading ${modelState.file.name} in this browser...`;
  status.dataset.state = '';
  try {
    const ort = await loadOnnxRuntime();
    const modelBytes = new Uint8Array(await modelState.file.arrayBuffer());
    const session = await ort.InferenceSession.create(modelBytes, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all'
    });
    if (!session.inputNames?.length || !session.outputNames?.length) {
      await session.release();
      throw new Error(`${model} ONNX file must have at least one input and output tensor.`);
    }

    modelState.session = session;
    status.textContent = `${modelState.file.name} loaded; running with WebAssembly`;
    status.dataset.state = 'ready';
    const inference = document.getElementById(`benchInf-${model}`);
    const benchmarkStatus = document.getElementById(`benchStatus-${model}`);
    if (inference) inference.textContent = 'Loaded; waiting for frame';
    if (benchmarkStatus) benchmarkStatus.textContent = 'ONNX loaded';
    if (selectedModel === model) updateSelectedModel(model);
    return session;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    status.textContent = `Could not load ONNX model: ${message}`;
    status.dataset.state = 'error';
    throw new Error(`Could not load the ${model} ONNX model: ${message}`, { cause: error });
  }
}

function getYoloConfig(model) {
  const labelsText = document.getElementById(`${model}Labels`).value;
  const classNames = labelsText.split(',').map(label => label.trim()).filter(Boolean);
  const targetText = document.getElementById(`${model}DrowsyClasses`).value.trim();
  const targetIdValues = targetText ? targetText.split(',').map(value => value.trim()) : [];
  if (targetIdValues.some(value => !/^\d+$/.test(value))) {
    throw new Error('Drowsy class IDs must be non-negative integers separated by commas.');
  }
  const targetClassIds = targetIdValues.map(Number);
  if (targetClassIds.some(value => !Number.isInteger(value) || value < 0)) {
    throw new Error('Drowsy class IDs must be non-negative integers separated by commas.');
  }

  const threshold = Number(document.getElementById(`${model}Confidence`).value);
  if (!Number.isFinite(threshold) || threshold < 0.01 || threshold > 0.99) {
    throw new Error('YOLO score threshold must be between 0.01 and 0.99.');
  }
  const task = document.getElementById(`${model}Task`).value;
  if (task === 'classification') {
    if (classNames.length < 2) {
      throw new Error('Enter all classifier labels in model order, separated by commas.');
    }
    if (targetClassIds.some(classId => classId >= classNames.length)) {
      throw new Error(`Target class IDs must be less than the classifier class count (${classNames.length}).`);
    }
  }
  return { classNames, targetClassIds, threshold, task };
}

function prepareYoloInput(video, size, task) {
  if (!yoloInputCanvas) {
    yoloInputCanvas = document.createElement('canvas');
  }
  if (yoloInputCanvas.width !== size || yoloInputCanvas.height !== size) {
    yoloInputCanvas.width = size;
    yoloInputCanvas.height = size;
    yoloInputContext = yoloInputCanvas.getContext('2d', { willReadFrequently: true });
  }
  const inputContext = yoloInputContext;
  let scale = size / Math.max(video.videoWidth, video.videoHeight);
  let padX = 0;
  let padY = 0;
  let imageWidth = size;
  let imageHeight = size;
  if (task === 'classification') {
    const cropSize = Math.min(video.videoWidth, video.videoHeight);
    const cropX = (video.videoWidth - cropSize) / 2;
    const cropY = (video.videoHeight - cropSize) / 2;
    inputContext.drawImage(video, cropX, cropY, cropSize, cropSize, 0, 0, size, size);
  } else {
    scale = Math.min(size / video.videoWidth, size / video.videoHeight);
    imageWidth = Math.round(video.videoWidth * scale);
    imageHeight = Math.round(video.videoHeight * scale);
    padX = Math.floor((size - imageWidth) / 2);
    padY = Math.floor((size - imageHeight) / 2);
    inputContext.fillStyle = 'rgb(114, 114, 114)';
    inputContext.fillRect(0, 0, size, size);
    inputContext.drawImage(video, padX, padY, imageWidth, imageHeight);
  }

  const pixels = inputContext.getImageData(0, 0, size, size).data;
  const planeSize = size * size;
  const tensorData = new Float32Array(planeSize * 3);
  for (let pixel = 0; pixel < planeSize; pixel++) {
    tensorData[pixel] = pixels[pixel * 4] / 255;
    tensorData[planeSize + pixel] = pixels[pixel * 4 + 1] / 255;
    tensorData[planeSize * 2 + pixel] = pixels[pixel * 4 + 2] / 255;
  }
  return {
    data: tensorData,
    scale,
    padX,
    padY,
    imageWidth: video.videoWidth,
    imageHeight: video.videoHeight,
    inputSize: size
  };
}

function decodeYoloOutput(output, model, transform) {
  const dimensions = output.dims;
  if (!Array.isArray(dimensions) || dimensions.length < 2 || dimensions.length > 3) {
    throw new Error('Unsupported YOLO output. Expected a raw detection tensor or end-to-end NMS tensor.');
  }
  if (dimensions.length === 3 && dimensions[0] !== 1) {
    throw new Error('YOLO ONNX export must use a batch size of 1.');
  }
  const config = getYoloConfig(model);
  const data = output.data;
  const rawRows = dimensions.length === 2 ? dimensions[0] : dimensions[1];
  const rawColumns = dimensions.length === 2 ? dimensions[1] : dimensions[2];
  const channelsFirst = rawRows <= 256 && rawColumns > 256;
  const rowCount = channelsFirst ? rawColumns : rawRows;
  const columnCount = channelsFirst ? rawRows : rawColumns;
  const valueAt = (row, column) => channelsFirst
    ? data[column * rowCount + row]
    : data[row * columnCount + column];

  const nmsOutput = !channelsFirst && columnCount === 6 && rowCount <= 1_000;
  const candidates = [];
  if (nmsOutput) {
    for (let row = 0; row < rowCount; row++) {
      const score = valueAt(row, 4);
      if (!Number.isFinite(score) || score < config.threshold) continue;
      const classId = Math.round(valueAt(row, 5));
      if (classId < 0 || (config.classNames.length && classId >= config.classNames.length)) {
        throw new Error('YOLO output class IDs do not match the configured class labels.');
      }
      candidates.push({
        x1: valueAt(row, 0), y1: valueAt(row, 1),
        x2: valueAt(row, 2), y2: valueAt(row, 3),
        score, classId
      });
    }
  } else {
    let hasObjectness = false;
    if (config.classNames.length) {
      if (columnCount === config.classNames.length + 5) hasObjectness = true;
      else if (columnCount !== config.classNames.length + 4) {
        throw new Error(
          `This detection model has ${columnCount - 4} or ${columnCount - 5} classes, but ${config.classNames.length} labels are configured. ` +
          'Use labels matching the detector, or select Eye-state classification and load the trained eye-state classifier. ' +
          'The standard yolov8n.onnx COCO model does not detect eye states.'
        );
      }
    }
    const classCount = columnCount - (hasObjectness ? 5 : 4);
    if (classCount < 1) throw new Error('YOLO output does not contain any class scores.');
    if (config.targetClassIds.some(classId => classId >= classCount)) {
      throw new Error(`Drowsy class IDs must be less than the model class count (${classCount}).`);
    }
    for (let row = 0; row < rowCount; row++) {
      let classId = 0;
      let classScore = -Infinity;
      for (let classIndex = 0; classIndex < classCount; classIndex++) {
        const score = valueAt(row, 4 + (hasObjectness ? 1 : 0) + classIndex);
        if (score > classScore) {
          classScore = score;
          classId = classIndex;
        }
      }
      const objectness = hasObjectness ? valueAt(row, 4) : 1;
      const score = classScore * objectness;
      if (!Number.isFinite(score) || score < config.threshold) continue;
      const centerX = valueAt(row, 0);
      const centerY = valueAt(row, 1);
      const width = valueAt(row, 2);
      const height = valueAt(row, 3);
      candidates.push({
        x1: centerX - width / 2,
        y1: centerY - height / 2,
        x2: centerX + width / 2,
        y2: centerY + height / 2,
        score,
        classId
      });
    }
  }

  const boxes = candidates.map(candidate => {
    const normalized = Math.max(
      Math.abs(candidate.x1), Math.abs(candidate.x2),
      Math.abs(candidate.y1), Math.abs(candidate.y2)
    ) <= 1.5;
    const xScale = normalized ? YOLO_INPUT_SIZE : 1;
    const yScale = normalized ? YOLO_INPUT_SIZE : 1;
    const x1 = Math.max(0, Math.min(transform.imageWidth,
      (candidate.x1 * xScale - transform.padX) / transform.scale));
    const y1 = Math.max(0, Math.min(transform.imageHeight,
      (candidate.y1 * yScale - transform.padY) / transform.scale));
    const x2 = Math.max(0, Math.min(transform.imageWidth,
      (candidate.x2 * xScale - transform.padX) / transform.scale));
    const y2 = Math.max(0, Math.min(transform.imageHeight,
      (candidate.y2 * yScale - transform.padY) / transform.scale));
    return {
      x1, y1, x2, y2,
      score: candidate.score,
      classId: candidate.classId,
      label: config.classNames[candidate.classId] || `class ${candidate.classId}`
    };
  }).filter(box => box.x2 > box.x1 && box.y2 > box.y1)
    .sort((left, right) => right.score - left.score);
  return suppressOverlappingBoxes(boxes).slice(0, 100);
}

function suppressOverlappingBoxes(boxes) {
  const selected = [];
  for (const candidate of boxes) {
    const area = (candidate.x2 - candidate.x1) * (candidate.y2 - candidate.y1);
    const overlaps = selected.some(existing => {
      if (existing.classId !== candidate.classId) return false;
      const width = Math.max(0, Math.min(candidate.x2, existing.x2) - Math.max(candidate.x1, existing.x1));
      const height = Math.max(0, Math.min(candidate.y2, existing.y2) - Math.max(candidate.y1, existing.y1));
      const intersection = width * height;
      const existingArea = (existing.x2 - existing.x1) * (existing.y2 - existing.y1);
      return intersection / (area + existingArea - intersection) > 0.45;
    });
    if (!overlaps) selected.push(candidate);
  }
  return selected;
}

async function runYoloInference(model, video) {
  const session = yoloModels[model].session;
  const config = getYoloConfig(model);
  const inputSize = config.task === 'classification' ? 224 : YOLO_INPUT_SIZE;
  const input = prepareYoloInput(video, inputSize, config.task);
  const tensor = new window.ort.Tensor('float32', input.data, [1, 3, inputSize, inputSize]);
  const start = performance.now();
  const outputs = await session.run({ [session.inputNames[0]]: tensor });
  const elapsed = performance.now() - start;
  const output = outputs[session.outputNames[0]];
  if (!output) throw new Error('YOLO model output tensor is missing.');
  if (config.task === 'classification') {
    if (output.dims.length !== 2 || output.dims[0] !== 1 || output.dims[1] !== config.classNames.length) {
      throw new Error(`Classifier output must have shape [1, ${config.classNames.length}] to match the configured labels.`);
    }
    const values = Array.from(output.data);
    const total = values.reduce((sum, value) => sum + value, 0);
    const areProbabilities = values.every(value => value >= 0 && value <= 1) && Math.abs(total - 1) < 0.02;
    const scores = areProbabilities
      ? values
      : (() => {
          const max = Math.max(...values);
          const exps = values.map(value => Math.exp(value - max));
          const sum = exps.reduce((accumulator, value) => accumulator + value, 0);
          return exps.map(value => value / sum);
        })();
    const classId = scores.indexOf(Math.max(...scores));
    return {
      classification: {
        classId,
        label: config.classNames[classId],
        score: scores[classId],
      },
      inferenceMs: elapsed,
    };
  }
  return {
    detections: decodeYoloOutput(output, model, input),
    inferenceMs: elapsed
  };
}

function drawYoloDetections(detections, model) {
  const color = model === 'yolo11' ? '#00f2fe' : '#f97316';
  ctx.save();
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = Math.max(2, canvas.width / 320);
  ctx.font = `${Math.max(14, canvas.width / 45)}px Inter, sans-serif`;
  detections.forEach(detection => {
    const width = detection.x2 - detection.x1;
    const height = detection.y2 - detection.y1;
    const text = `${detection.label} ${(detection.score * 100).toFixed(1)}%`;
    const textWidth = ctx.measureText(text).width;
    const labelY = Math.max(0, detection.y1 - 24);
    ctx.strokeRect(detection.x1, detection.y1, width, height);
    ctx.fillRect(detection.x1, labelY, textWidth + 10, 24);
    ctx.fillStyle = '#07101b';
    ctx.fillText(text, detection.x1 + 5, labelY + 17);
    ctx.fillStyle = color;
  });
  ctx.restore();
}

function updateYoloDetections(model, detections) {
  yoloDetections = detections;
  const confidenceEl = document.getElementById('yoloConfValue');
  const inferenceEl = document.getElementById(`benchInf-${model}`);
  const fpsEl = document.getElementById(`benchFps-${model}`);
  const statusEl = document.getElementById(`benchStatus-${model}`);
  const topDetection = detections[0];
  if (confidenceEl) {
    confidenceEl.textContent = topDetection
      ? `${(topDetection.score * 100).toFixed(1)}% · ${topDetection.label}`
      : 'No detections';
  }
  if (inferenceEl) inferenceEl.textContent = `${yoloInferenceMs.toFixed(1)} ms`;
  if (fpsEl) fpsEl.textContent = `${currentFps} fps`;
  const config = getYoloConfig(model);
  const targetDetection = detections.find(detection =>
    config.targetClassIds.includes(detection.classId)
  );
  if (statusEl) {
    statusEl.textContent = targetDetection
      ? `${detections.length} ${detections.length === 1 ? 'detection' : 'detections'} · target class ${targetDetection.classId}`
      : `${detections.length} ${detections.length === 1 ? 'detection' : 'detections'}`;
  }

  if (!targetDetection) {
    yoloTargetAlerted = false;
    return;
  }
  if (yoloTargetAlerted) return;
  yoloTargetAlerted = true;
  triggerAlert({
    message: `The YOLO model detected ${targetDetection.label}.`,
    voiceText: `Drowsiness class detected: ${targetDetection.label}.`,
    soundType: 'closure',
    alertType: 'yolo'
  });
}

async function startRealDetection(runId) {
  try {
    if (selectedModel === 'mediapipe') {
      await loadSelectedModel();
      if (runId !== detectionRunId) {
        if (mediaPipeLandmarker) mediaPipeLandmarker.close();
        mediaPipeLandmarker = null;
        return;
      }
    } else {
      document.getElementById('loadingMessage').textContent = `Loading ${selectedModel} ONNX model...`;
      await loadYoloModel(selectedModel);
      if (runId !== detectionRunId) return;
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
    setStatus(selectedModel === 'mediapipe' ? 'MediaPipe active' : `${selectedModel} ONNX active`, true);
    lastVideoTime = -1;
    detectRealFrame(runId);
  } catch (error) {
    console.error('Face detection could not start:', error);
    if (runId !== detectionRunId) return;
    stopDetection();
    document.getElementById('loadingMessage').textContent = `${error.message}. Check network/camera permissions and try again.`;
    document.getElementById('videoLoading').classList.remove('hidden');
    setStatus('Detection unavailable', false);
  }
}

async function detectRealFrame(runId) {
  if (!isRunning || !modelsLoaded || runId !== detectionRunId) return;
  // ── CONCURRENCY GUARD: Eliminates video lag by skipping frames while inference is busy ──
  if (isProcessingFrame) {
    scheduleDetectionFrame(document.getElementById('video'), runId);
    return;
  }
  const video = document.getElementById('video');
  if (video.currentTime === lastVideoTime) {
    scheduleDetectionFrame(video, runId);
    return;
  }
  lastVideoTime = video.currentTime;
  isProcessingFrame = true;
  try {
    if (selectedModel === 'yolov8' || selectedModel === 'yolo11') {
      const yoloResult = await runYoloInference(selectedModel, video);
      yoloInferenceMs = yoloResult.inferenceMs;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      if (yoloResult.classification) {
        updateYoloClassification(selectedModel, yoloResult.classification);
        drawYoloClassification(yoloResult.classification, selectedModel);
        processYoloClassificationSample(selectedModel, yoloResult.classification);
      } else {
        updateYoloDetections(selectedModel, yoloResult.detections);
        drawYoloDetections(yoloResult.detections, selectedModel);
        processYoloSample(selectedModel, yoloResult.detections);
      }
    } else {
      let ear = null;
      let landmarks = null;
      const inferenceStart = performance.now();
      const result = mediaPipeLandmarker.detectForVideo(video, inferenceStart);
      lastInferenceMs = performance.now() - inferenceStart;
      landmarks = result.faceLandmarks?.[0] || null;
      if (landmarks) {
        const leftEye  = [33, 160, 158, 133, 153, 144].map(i => landmarks[i]);
        const rightEye = [362, 385, 387, 263, 373, 380].map(i => landmarks[i]);
        ear = (computeEAR(leftEye) + computeEAR(rightEye)) / 2;
        updateBlinkCount(ear);
        updateGazeAttention(landmarks, ear);
        updateMARMetric(computeMAR(landmarks));
      } else {
        resetFaceNotDetected();
      }

      const infEl = document.getElementById('inferenceValue');
      if (infEl) infEl.textContent = `${lastInferenceMs.toFixed(1)} ms`;
      const benchInfEl = document.getElementById('benchInf-mediapipe');
      const benchFpsEl = document.getElementById('benchFps-mediapipe');
      if (benchInfEl) benchInfEl.textContent = `${lastInferenceMs.toFixed(1)} ms`;
      if (benchFpsEl) benchFpsEl.textContent = `${currentFps} fps`;

      const faceDetected = ear !== null && Number.isFinite(ear);
      if (faceDetected) drawModelMesh(landmarks, selectedModel);
      else ctx.clearRect(0, 0, canvas.width, canvas.height);
      processDetectionSample(faceDetected ? ear : null, faceDetected);
    }
  } catch (error) {
    console.error('Detection frame failed:', error);
    if (isRunning) {
      const message = error instanceof Error ? error.message : String(error);
      stopDetection();
      document.getElementById('loadingMessage').textContent = `Detection failed: ${message}`;
      document.getElementById('videoLoading').classList.remove('hidden');
      setStatus('Detection error', false);
    }
  } finally {
    if (runId === detectionRunId) calcFPS();
    isProcessingFrame = false;
    scheduleDetectionFrame(video, runId);
  }
}

function processYoloSample(model, detections) {
  const now = Date.now();
  const config = getYoloConfig(model);
  const targetDetection = detections.find(detection => config.targetClassIds.includes(detection.classId));
  const inferenceEl = document.getElementById('inferenceValue');
  const detectionCount = document.getElementById('faceDetected');
  const headAlignment = document.getElementById('headAlignmentValue');
  if (inferenceEl) inferenceEl.textContent = `${yoloInferenceMs.toFixed(1)} ms`;
  if (detectionCount) detectionCount.textContent = String(detections.length);
  if (headAlignment) headAlignment.textContent = 'Unavailable';
  recordEvaluationSample(0, detections.length > 0);
  document.getElementById('evaluationScore').textContent = 'LIVE';
  document.getElementById('evaluationStatus').textContent =
    config.targetClassIds.length
      ? `Alerts on the first detection of class IDs: ${config.targetClassIds.join(', ')}.`
      : 'YOLO object detection only; no drowsiness alert classes are configured.';
  document.getElementById('faceCoverage').textContent =
    evaluation.frames ? `${((evaluation.faceFrames / evaluation.frames) * 100).toFixed(1)}%` : '--';
  document.getElementById('averageEar').textContent = targetDetection
    ? `${(targetDetection.score * 100).toFixed(1)}%`
    : '--';
  document.getElementById('alertRate').textContent =
    sessionStart ? (alertCount / Math.max((now - sessionStart) / 60000, 1 / 60)).toFixed(1) : '0.0';
  document.getElementById('framesSampled').textContent = String(evaluation.frames);
  document.getElementById('perclosValue').textContent = 'Unavailable in YOLO-only mode';
  document.getElementById('earValue').textContent = 'Unavailable';
  document.getElementById('marValue').textContent = 'Unavailable';
  document.getElementById('closureDuration').textContent = 'Unavailable';
  document.getElementById('blinkCount').textContent = '—';
  document.getElementById('yawnCount').textContent = '—';
  document.getElementById('longClosureCount').textContent = '—';
  if (now - lastLoggedAt >= LOG_INTERVAL_MS) {
    logEvent('sample', {
      selectedMode: selectedModel,
      inferenceBackend: `${model}-onnx`,
      inferenceMs: Number(yoloInferenceMs.toFixed(2)),
      fps: currentFps,
      detectionCount: detections.length,
      targetClassIds: config.targetClassIds.join(','),
      targetConfidence: targetDetection ? Number(targetDetection.score.toFixed(4)) : '',
      targetClass: targetDetection?.label || ''
    });
    lastLoggedAt = now;
  }
}

function updateYoloClassification(model, prediction) {
  const confidenceEl = document.getElementById('yoloConfValue');
  const inferenceEl = document.getElementById(`benchInf-${model}`);
  const fpsEl = document.getElementById(`benchFps-${model}`);
  const statusEl = document.getElementById(`benchStatus-${model}`);
  const config = getYoloConfig(model);
  const isTarget = config.targetClassIds.includes(prediction.classId);
  if (confidenceEl) confidenceEl.textContent = `${(prediction.score * 100).toFixed(1)}% · ${prediction.label}`;
  if (inferenceEl) inferenceEl.textContent = `${yoloInferenceMs.toFixed(1)} ms`;
  if (fpsEl) fpsEl.textContent = `${currentFps} fps`;
  if (statusEl) statusEl.textContent = `${prediction.label} · ${(prediction.score * 100).toFixed(1)}%`;
  if (!isTarget) yoloTargetAlerted = false;
}

function drawYoloClassification(prediction, model) {
  const color = model === 'yolo11' ? '#00f2fe' : '#f97316';
  const text = `${prediction.label} ${(prediction.score * 100).toFixed(1)}%`;
  ctx.save();
  ctx.font = `${Math.max(14, canvas.width / 45)}px Inter, sans-serif`;
  const width = ctx.measureText(text).width + 20;
  ctx.fillStyle = color;
  ctx.fillRect(8, 8, width, 30);
  ctx.fillStyle = '#07101b';
  ctx.fillText(text, 18, 29);
  ctx.restore();
}

function processYoloClassificationSample(model, prediction) {
  const now = Date.now();
  const config = getYoloConfig(model);
  const isTarget = config.targetClassIds.includes(prediction.classId)
    && prediction.score >= config.threshold;
  const closureMs = updateClosureMetrics(isTarget);
  if (isTarget && closureMs >= sens.alertMs && !yoloTargetAlerted) {
    yoloTargetAlerted = true;
    triggerAlert({
      message: `Eye-state classifier reports ${prediction.label}.`,
      voiceText: `Please open your eyes. The model reports ${prediction.label}.`,
      alertType: 'closure',
      soundType: 'closure'
    });
  }
  if (!isTarget) yoloTargetAlerted = false;

  if (!blinkTrackingClosed && isTarget) {
    blinkTrackingClosed = true;
    blinkTrackingStart = performance.now();
  } else if (blinkTrackingClosed && !isTarget) {
    const duration = performance.now() - blinkTrackingStart;
    if (duration >= 40 && duration < 500) {
      blinkCount++;
      document.getElementById('blinkCount').textContent = String(blinkCount);
    }
    blinkTrackingClosed = false;
    blinkTrackingStart = null;
  }

  const perclosResult = updatePerclos(true, isTarget, now);
  const elapsedSeconds = Math.min((performance.now() - lastScoreUpdate) / 1000, 0.25);
  drowsinessScore = isTarget
    ? Math.min(100, drowsinessScore + elapsedSeconds * 12)
    : Math.max(0, drowsinessScore - elapsedSeconds * 8);
  lastScoreUpdate = performance.now();
  updateGauge(drowsinessScore);
  applyDrowsyState(drowsinessScore);

  document.getElementById('faceDetected').textContent = 'Frame classified';
  document.getElementById('headAlignmentValue').textContent = 'Unavailable';
  document.getElementById('earValue').textContent = 'Not measured';
  document.getElementById('earStatus').textContent = `${prediction.label} (${(prediction.score * 100).toFixed(1)}%)`;
  document.getElementById('earBar').style.width = `${(prediction.score * 100).toFixed(1)}%`;
  document.getElementById('perclosValue').textContent = perclosResult.value === null
    ? 'Insufficient samples'
    : `${perclosResult.value.toFixed(1)}% · experimental`;
  document.getElementById('perclosCoverage').textContent =
    `Classifier frame coverage ${Math.min(100, perclosResult.coverage).toFixed(0)}% · not face tracking`;
  document.getElementById('longClosureCount').textContent = String(longClosureCount);
  document.getElementById('blinkCount').textContent = String(blinkCount);
  document.getElementById('marValue').textContent = 'Unavailable';
  recordEvaluationSample(0, true);
  document.getElementById('evaluationScore').textContent = 'LIVE';
  document.getElementById('evaluationStatus').textContent =
    'Experimental full-frame classifier. Eye-state estimates only; no gaze or face detection.';
  document.getElementById('faceCoverage').textContent =
    `${((evaluation.faceFrames / Math.max(evaluation.frames, 1)) * 100).toFixed(1)}%`;
  document.getElementById('averageEar').textContent = `${(prediction.score * 100).toFixed(1)}%`;
  document.getElementById('alertRate').textContent =
    sessionStart ? (alertCount / Math.max((now - sessionStart) / 60000, 1 / 60)).toFixed(1) : '0.0';
  document.getElementById('framesSampled').textContent = String(evaluation.frames);

  if (now - lastLoggedAt >= LOG_INTERVAL_MS) {
    logEvent('sample', {
      selectedMode: model,
      inferenceBackend: `${model}-onnx-classifier`,
      inferenceMs: Number(yoloInferenceMs.toFixed(2)),
      fps: currentFps,
      predictedClass: prediction.label,
      predictedClassId: prediction.classId,
      predictedScore: Number(prediction.score.toFixed(4)),
      experimentalPerclos60s: perclosResult.value === null ? '' : Number(perclosResult.value.toFixed(2))
    });
    lastLoggedAt = now;
  }
}

function resetFaceNotDetected() {
  blinkTrackingClosed = false;
  blinkTrackingStart = null;
  gazeAwayStart = null;
  gazeAwayAlerted = false;
  marBuffer = [];
  currentMAR = 0.20;
  yawnFrames = 0;
  yawnEpisodeActive = false;
  yawnAlerted = false;
  const marValueEl = document.getElementById('marValue');
  if (marValueEl) marValueEl.textContent = '—';
  const marBarEl = document.getElementById('marBar');
  if (marBarEl) {
    marBarEl.style.width = '0%';
    marBarEl.className = 'metric-bar';
  }
  const marStatusEl = document.getElementById('marStatus');
  if (marStatusEl) marStatusEl.textContent = 'Face not tracked';
  gazeAlignmentBuffer = [];
  const gazeStatusEl = document.getElementById('gazeAlignmentStatus');
  if (gazeStatusEl) { gazeStatusEl.textContent = 'Face not detected'; gazeStatusEl.style.color = ''; }
  const gazeValEl = document.getElementById('gazeAlignmentValue');
  if (gazeValEl) gazeValEl.textContent = '—';
  const gazeBarEl = document.getElementById('gazeAlignmentBar');
  if (gazeBarEl) { gazeBarEl.style.width = '0%'; gazeBarEl.className = 'metric-bar'; }
  const headValEl = document.getElementById('headAlignmentValue');
  if (headValEl) headValEl.textContent = 'No face';
}

function scheduleDetectionFrame(video, runId) {
  if (!isRunning || runId !== detectionRunId) return;
  if (typeof video.requestVideoFrameCallback === 'function') {
    videoFrameCallbackId = video.requestVideoFrameCallback(() => {
      videoFrameCallbackId = null;
      detectRealFrame(runId);
    });
  } else {
    animFrameId = requestAnimationFrame(() => {
      animFrameId = null;
      detectRealFrame(runId);
    });
  }
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
    if (closed && closureMs >= sens.alertMs) {
      triggerAlert({
        message: 'Please open your eyes.',
        voiceText: 'Please open your eyes.',
        alertType: 'closure',
        soundType: 'closure'
      });
    }
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
        message: 'Please face the camera to resume tracking.',
        voiceText: 'Please face the camera.',
        soundType: 'gaze',
        alertType: 'noface'
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
      selectedMode: selectedModel,
      inferenceBackend: selectedModel === 'mediapipe' ? 'mediapipe-face-landmarker' : `${selectedModel}-onnx`,
      inferenceBackendVersion: selectedModel === 'mediapipe' ? MEDIAPIPE_VERSION : '',
      perclos60s: perclos === null ? '' : Number(perclos.toFixed(2)),
      trackingCoverage60s: Number(perclosResult.coverage.toFixed(2)),
      inferenceMs: Number(lastInferenceMs.toFixed(2)),
      fps: currentFps,
      mar: faceDetected ? Number(currentMAR.toFixed(3)) : '',
      yawnThreshold: MAR_THRESHOLD,
      yoloInferenceMs: selectedModel === 'mediapipe' ? '' : Number(yoloInferenceMs.toFixed(2)),
      yoloDetectionCount: selectedModel === 'mediapipe' ? '' : yoloDetections.length,
      yoloConfidence: selectedModel === 'mediapipe' || !yoloDetections[0]
        ? ''
        : Number(yoloDetections[0].score.toFixed(4)),
      yoloClass: selectedModel === 'mediapipe' ? '' : (yoloDetections[0]?.label || ''),
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
      message: 'Please look straight ahead.',
      voiceText: 'Please look straight ahead.',
      soundType: 'gaze',
      alertType: 'gaze'
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
  voiceSettings = { enabled: vEnabled, volume: vVolume };
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
    if (vs) {
      voiceSettings = {
        enabled: typeof vs.enabled === 'boolean' ? vs.enabled : voiceSettings.enabled,
        volume: Number.isFinite(Number(vs.volume)) ? Number(vs.volume) : voiceSettings.volume
      };
    }
  } catch (e) { /* ignore */ }
}

// -------------------------
// Spoken alert prompts
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

function stopVoice() {
  if (window.speechSynthesis) window.speechSynthesis.cancel();
}

// -------------------------
// EAR sparkline
// -------------------------
function updateSparkline(ear) {
  const c = document.getElementById('earSpark');
  if (!c) return;
  earHistory.push(ear);
  if (earHistory.length > 100) earHistory.shift();
  const now = performance.now();
  if (now - lastSparklineUpdate < 100) return;
  lastSparklineUpdate = now;
  const ctx2 = c.getContext('2d');
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

// ── Mouth Aspect Ratio (MAR) for Yawn Detection ───────
// Uses MediaPipe mouth landmarks: outer lips top/bottom + left/right corners
function computeMAR(landmarks) {
  // Mouth top: 13, bottom: 14 (inner), corners: 61 (left), 291 (right)
  // Use a richer set for MAR: vertical distances / horizontal width
  const topOuter    = landmarks[0]  || landmarks[13];   // upper lip top
  const bottomOuter = landmarks[17] || landmarks[14];   // lower lip bottom
  const topInner    = landmarks[13];
  const bottomInner = landmarks[14];
  const leftCorner  = landmarks[61];
  const rightCorner = landmarks[291];
  if (!topOuter || !bottomOuter || !leftCorner || !rightCorner) return 0.2;
  const A = dist(topOuter, bottomOuter);
  const B = dist(topInner, bottomInner);
  const C = dist(leftCorner, rightCorner);
  return C > 0 ? (A + B) / (2.0 * C) : 0.2;
}

// ── Update MAR UI ──────────────────────────────────────
function updateMARMetric(mar) {
  marBuffer.push(mar);
  if (marBuffer.length > 6) marBuffer.shift();
  currentMAR = marBuffer.reduce((a, b) => a + b, 0) / marBuffer.length;

  const marEl  = document.getElementById('marValue');
  const barEl  = document.getElementById('marBar');
  const statEl = document.getElementById('marStatus');
  if (!marEl) return;

  marEl.textContent = currentMAR.toFixed(3);
  const pct = Math.min(100, (currentMAR / 1.0) * 100);
  if (barEl) {
    barEl.style.width = pct + '%';
    barEl.className   = currentMAR >= MAR_THRESHOLD ? 'metric-bar danger-bar' : 'metric-bar';
  }
  const isYawning = currentMAR >= MAR_THRESHOLD;
  if (statEl) {
    statEl.textContent = isYawning ? 'YAWN DETECTED — Fatigue Indicator' : 'Mouth closed · Optimal';
    statEl.style.color = isYawning ? 'var(--danger)' : '';
  }

  if (isYawning) {
    yawnFrames++;
    if (!yawnEpisodeActive) {
      yawnEpisodeActive = true;
      yawnCount++;
      const yawnCountEl = document.getElementById('yawnCount');
      if (yawnCountEl) yawnCountEl.textContent = yawnCount;
    }
    if (yawnFrames >= 20 && !yawnAlerted) {
      triggerAlert({
        message: 'Yawning detected. Take a break.',
        voiceText: 'Yawning detected. Please take a break.',
        soundType: 'gaze',
        alertType: 'yawn'
      });
      yawnAlerted = true;
    }
  } else {
    yawnFrames = 0;
    yawnEpisodeActive = false;
    yawnAlerted = false;
  }
}

// ── Unified Draw Model Mesh ─────────────────────────────
// Dispatches to model-specific rendering with theme-matched colours
function drawModelMesh(landmarks, model) {
  const isDrowsy = drowsinessScore > 50;
  let accentColor, dangerColor, dotColor;

  if (model === 'yolo11') {
    accentColor = isDrowsy ? 'rgba(239,68,68,0.65)'  : 'rgba(0,242,254,0.55)';
    dangerColor = isDrowsy ? 'rgba(239,68,68,0.65)'  : 'rgba(0,242,254,0.55)';
    dotColor    = isDrowsy ? 'rgba(239,68,68,0.80)'  : 'rgba(0,242,254,0.80)';
  } else if (model === 'yolov8') {
    accentColor = isDrowsy ? 'rgba(239,68,68,0.65)'  : 'rgba(249,115,22,0.60)';
    dangerColor = isDrowsy ? 'rgba(239,68,68,0.65)'  : 'rgba(249,115,22,0.60)';
    dotColor    = isDrowsy ? 'rgba(239,68,68,0.80)'  : 'rgba(249,115,22,0.80)';
  } else {
    accentColor = isDrowsy ? 'rgba(239,68,68,0.60)'  : 'rgba(0,212,170,0.50)';
    dangerColor = isDrowsy ? 'rgba(239,68,68,0.60)'  : 'rgba(0,212,170,0.50)';
    dotColor    = isDrowsy ? 'rgba(239,68,68,0.75)'  : 'rgba(0,212,170,0.75)';
  }

  let minX = canvas.width, minY = canvas.height, maxX = 0, maxY = 0;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // ── BATCHED landmark dots: single path per colour → massive GPU savings ──
  ctx.fillStyle = dotColor;
  ctx.beginPath();
  for (let i = 0; i < landmarks.length; i += (model === 'mediapipe' ? 6 : 4)) {
    const p = landmarks[i];
    const x = p.x * canvas.width;
    const y = p.y * canvas.height;
    minX = Math.min(minX, x); minY = Math.min(minY, y);
    maxX = Math.max(maxX, x); maxY = Math.max(maxY, y);
    ctx.moveTo(x + 1.2, y);
    ctx.arc(x, y, 1.2, 0, 6.2832);
  }
  ctx.fill();

  // Face bounding box
  ctx.strokeStyle = accentColor;
  ctx.lineWidth   = 2;
  if (typeof ctx.roundRect === 'function') {
    ctx.beginPath();
    ctx.roundRect(minX, minY, maxX - minX, maxY - minY, 8);
    ctx.stroke();
  } else {
    ctx.strokeRect(minX, minY, maxX - minX, maxY - minY);
  }

  // Eye contours (colour-matched)
  drawContourColoured([33,160,158,133,153,144].map(i => ({
    x: landmarks[i].x * canvas.width, y: landmarks[i].y * canvas.height
  })), isDrowsy, model);
  drawContourColoured([362,385,387,263,373,380].map(i => ({
    x: landmarks[i].x * canvas.width, y: landmarks[i].y * canvas.height
  })), isDrowsy, model);

  // YOLO modes add a mouth contour for the MAR overlay.
  if (model === 'yolov8' || model === 'yolo11') {
    drawContourColoured([61,185,40,39,37,0,267,269,270,409,291].map(i => ({
      x: landmarks[i].x * canvas.width, y: landmarks[i].y * canvas.height
    })), isDrowsy, model);
  }

  // HUD badge (all models)
  drawHeadAlignmentHUD(model);
}

// Legacy wrapper — keeps MediaPipe path working unchanged
function drawMediaPipeMesh(landmarks) {
  drawModelMesh(landmarks, 'mediapipe');
}

function drawFaceMesh(landmarks, box) { /* legacy stub */ }

function drawContourColoured(points, danger, model) {
  let strokeCol, fillCol;
  if (model === 'yolo11') {
    strokeCol = danger ? 'rgba(239,68,68,0.9)' : 'rgba(0,242,254,0.9)';
    fillCol   = danger ? 'rgba(239,68,68,0.08)': 'rgba(0,242,254,0.08)';
  } else if (model === 'yolov8') {
    strokeCol = danger ? 'rgba(239,68,68,0.9)' : 'rgba(249,115,22,0.9)';
    fillCol   = danger ? 'rgba(239,68,68,0.08)': 'rgba(249,115,22,0.08)';
  } else {
    strokeCol = danger ? 'rgba(239,68,68,0.9)' : 'rgba(0,212,170,0.9)';
    fillCol   = danger ? 'rgba(239,68,68,0.08)': 'rgba(0,212,170,0.08)';
  }
  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);
  for (let i = 1; i < points.length; i++) ctx.lineTo(points[i].x, points[i].y);
  ctx.closePath();
  ctx.strokeStyle = strokeCol; ctx.lineWidth = 1.5; ctx.stroke();
  ctx.fillStyle   = fillCol;   ctx.fill();
}

function drawHeadAlignmentHUD(model) {
  ctx.save();
  ctx.translate(canvas.width, 0);
  ctx.scale(-1, 1);

  const isOk = currentHeadAlignment >= HEAD_ALIGNMENT_THRESHOLD;
  const badgeText = isOk
    ? `Head Alignment: ${currentHeadAlignment}% (≥70%) · Frontal`
    : `! Head Alignment: ${currentHeadAlignment}% (<70%) · ${gazeTurnDirection}`;

  // Pick accent colour per theme
  const okCol  = (model === 'yolo11') ? '#00f2fe' : (model === 'yolov8') ? '#f97316' : '#00d4aa';
  const badCol = '#ef4444';

  ctx.font = '600 12px Inter, sans-serif';
  const tw    = ctx.measureText(badgeText).width;
  const bW    = tw + 26, bH = 26, bX = 14, bY = 14;

  ctx.fillStyle = isOk ? 'rgba(6,13,24,0.85)' : 'rgba(239,68,68,0.90)';
  ctx.beginPath();
  if (typeof ctx.roundRect === 'function') ctx.roundRect(bX, bY, bW, bH, 8);
  else ctx.rect(bX, bY, bW, bH);
  ctx.fill();

  ctx.strokeStyle = isOk ? `${okCol}80` : 'rgba(255,255,255,0.7)';
  ctx.lineWidth   = 1;
  ctx.stroke();

  ctx.fillStyle = isOk ? okCol : '#fff';
  ctx.beginPath();
  ctx.arc(bX + 12, bY + 13, 4, 0, 6.2832);
  ctx.fill();

  ctx.fillStyle = '#fff';
  ctx.fillText(badgeText, bX + 22, bY + 17);
  ctx.restore();
}

function drawContour(points, danger) {
  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);
  points.forEach(p => ctx.lineTo(p.x, p.y));
  ctx.closePath();
  ctx.strokeStyle = danger ? 'rgba(239,68,68,0.9)' : 'rgba(0,212,170,0.9)';
  ctx.lineWidth   = 1.5;
  ctx.stroke();
  ctx.fillStyle   = danger ? 'rgba(239,68,68,0.08)' : 'rgba(0,212,170,0.08)';
  ctx.fill();
}

// ── UI Helpers ────────────────────────────────────────
function updateGauge(score) {
  const arc   = document.getElementById('gaugeArc');
  const total = 251;
  const offset = total - (score / 100) * total;
  arc.style.strokeDashoffset = offset;

  const percent = String(Math.round(score));
  const percentElement = document.getElementById('gaugePercent');
  if (percentElement.textContent !== percent) percentElement.textContent = percent;
  const lbl = document.getElementById('gaugeLabel');
  const label = score < 30 ? 'Alert' : score < 60 ? 'Drowsy' : 'Danger!';
  const color = score < 30 ? '#00d4aa' : score < 60 ? '#f59e0b' : '#ef4444';
  if (lbl.dataset.state !== label) {
    lbl.dataset.state = label;
    lbl.textContent = label;
    lbl.style.color = color;
  }
}

function applyDrowsyState(score) {
  const isDanger = score >= 60;
  const isWarning = !isDanger && score >= 30;
  document.body.classList.toggle('drowsy-danger', isDanger);
  document.body.classList.toggle('drowsy-warn', isWarning);

  const dot = document.getElementById('statusDot');
  const state = isDanger ? 'danger' : isWarning ? 'warn' : isRunning ? 'active' : 'inactive';
  if (dot.dataset.state === state) return;
  dot.dataset.state = state;
  if (isDanger) dot.className = 'pulse-dot danger';
  else if (isWarning) {
    dot.className = 'pulse-dot';
    dot.style.background = '#f59e0b';
  } else {
    dot.className = isRunning ? 'pulse-dot active' : 'pulse-dot';
    dot.style.background = '';
  }
}

function decayDrowsiness() {
  drowsinessScore = Math.max(0, drowsinessScore - 0.5);
  updateGauge(drowsinessScore);
}

function triggerAlert({
  message = 'Please open your eyes.',
  voiceText = message,
  soundType = 'closure',
  alertType = 'closure',
} = {}) {
  const now = Date.now();
  if (now - lastAlertTime < 7000) return;
  lastAlertTime = now;
  alertCount++;
  document.getElementById('alertCount').textContent = alertCount;

  // ── Official alert texts ──
  const textEl = document.getElementById('alertText');
  if (textEl) textEl.textContent = message;

  const headlineEl = document.getElementById('alertHeadline');
  if (headlineEl) {
    headlineEl.textContent =
      alertType === 'closure' ? 'EYE CLOSURE DETECTED' :
      alertType === 'yawn'    ? 'YAWN / FATIGUE EVENT' :
      alertType === 'gaze'    ? 'GAZE DEVIATION ALERT' :
      alertType === 'noface'  ? 'DRIVER NOT DETECTED'  :
      alertType === 'yolo'    ? 'YOLO CLASS DETECTED'  :
      'MONITORING ALERT';
  }

  // Fill diagnostics box
  const diagModel = document.getElementById('alertDiagModel');
  if (diagModel) diagModel.textContent = getModelLabel();
  const diagCause = document.getElementById('alertDiagCause');
  if (diagCause) {
    diagCause.textContent =
      alertType === 'closure' ? 'Prolonged Eye Closure' :
      alertType === 'yawn'    ? 'Yawn / Mouth Opening' :
      alertType === 'gaze'    ? 'Head / Gaze Turned Away' :
      alertType === 'yolo'    ? 'Configured YOLO drowsy class' :
      'Face Not Detected';
  }
  const diagRisk = document.getElementById('alertDiagRisk');
  if (diagRisk) {
    diagRisk.textContent = alertType === 'yolo'
      ? 'CLASS MATCH'
      : drowsinessScore >= 60 ? 'HIGH' : drowsinessScore >= 30 ? 'MODERATE' : 'LOW';
    diagRisk.style.color = alertType === 'yolo' || drowsinessScore >= 60
      ? '#ef4444'
      : drowsinessScore >= 30 ? '#f59e0b' : '#00d4aa';
  }

  logEvent('alert', {
    selectedMode: selectedModel,
    inferenceBackend: selectedModel === 'mediapipe' ? 'mediapipe-face-landmarker' : `${selectedModel}-onnx`,
    inferenceBackendVersion: selectedModel === 'mediapipe' ? MEDIAPIPE_VERSION : '',
    score: Number(drowsinessScore.toFixed(1)),
    closedFrames,
    closureDurationMs: eyeClosedStart ? Date.now() - eyeClosedStart : 0,
    perclos60s: document.getElementById('perclosValue').textContent,
    earThreshold: sens.ear,
    mar: Number(currentMAR.toFixed(3)),
    yoloConfidence: yoloDetections[0] ? Number(yoloDetections[0].score.toFixed(4)) : '',
    yoloClass: yoloDetections[0]?.label || '',
    alertType,
    headAlignment: currentHeadAlignment
  });

  // Reveal overlay
  document.getElementById('alertOverlay').classList.remove('hidden');

  // Play alert sound
  playAlertSound(soundType);
  closedFrames = 0;
  if (voiceSettings.enabled && voiceText) speak(voiceText, { volume: voiceSettings.volume });
}

function dismissAlert() {
  document.getElementById('alertOverlay').classList.add('hidden');
  drowsinessScore = Math.max(0, drowsinessScore - 30);
  stopVoice();
}

// Dramatic cinematic whoosh / anger impact boom (kept for MediaPipe compat, but unused in YOLO themes)
function playAngryRumbleSound() { /* removed – official alert uses no emoji sounds */ }

// Emoji eye-tracking stub (emoji stage hidden; no-op)
function initEmojiEyeTracking() {}
initEmojiEyeTracking();

function playAlertSound(type = 'closure') {
  try {
    const ctx2 = new (window.AudioContext || window.webkitAudioContext)();
    const now = ctx2.currentTime;

    if (type === 'gaze') {
      // Dual-tone attention chime
      const osc1 = ctx2.createOscillator(), gain1 = ctx2.createGain();
      osc1.connect(gain1); gain1.connect(ctx2.destination);
      osc1.type = 'triangle';
      osc1.frequency.setValueAtTime(880, now);
      osc1.frequency.exponentialRampToValueAtTime(659.25, now + 0.18);
      gain1.gain.setValueAtTime(0.35, now);
      gain1.gain.linearRampToValueAtTime(0.01, now + 0.22);
      osc1.start(now); osc1.stop(now + 0.22);

      const osc2 = ctx2.createOscillator(), gain2 = ctx2.createGain();
      osc2.connect(gain2); gain2.connect(ctx2.destination);
      osc2.type = 'triangle';
      osc2.frequency.setValueAtTime(1174.66, now + 0.16);
      osc2.frequency.exponentialRampToValueAtTime(880, now + 0.45);
      gain2.gain.setValueAtTime(0.35, now + 0.16);
      gain2.gain.linearRampToValueAtTime(0.01, now + 0.50);
      osc2.start(now + 0.16); osc2.stop(now + 0.50);
    } else {
      // Urgent rising siren for closure / drowsiness
      const osc = ctx2.createOscillator(), gain = ctx2.createGain();
      osc.connect(gain); gain.connect(ctx2.destination);
      osc.type = 'sawtooth';
      osc.frequency.setValueAtTime(440, now);
      osc.frequency.linearRampToValueAtTime(880, now + 0.3);
      gain.gain.setValueAtTime(0.4, now);
      gain.gain.linearRampToValueAtTime(0, now + 0.8);
      osc.start(now); osc.stop(now + 0.8);
    }
  } catch(e) { console.warn('AudioContext alert failed:', e); }
}

function updateStatCards(ear, closure, earSt, closeSt) {
  if (earSt)   document.getElementById('earStatus').textContent    = earSt;
  if (closeSt) document.getElementById('closureStatus').textContent = closeSt;
}

function setStatus(msg, active) {
  document.getElementById('statusText').textContent = msg;
  const dot = document.getElementById('statusDot');
  dot.className = active ? 'pulse-dot active' : 'pulse-dot';
  dot.dataset.state = active ? 'active' : 'inactive';
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
    if (selectedModel === 'mediapipe') {
      const benchmarkFps = document.getElementById('benchFps-mediapipe');
      if (benchmarkFps) benchmarkFps.textContent = `${currentFps} fps`;
      updateEvaluation();
    } else {
      const selectedFps = document.getElementById(`benchFps-${selectedModel}`);
      if (selectedFps) selectedFps.textContent = `${currentFps} fps`;
    }
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
