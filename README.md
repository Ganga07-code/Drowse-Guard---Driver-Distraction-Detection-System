# DrowseGuard

## Introduction

DrowseGuard is a browser-based research prototype that uses MediaPipe Face Landmarker to monitor eye closure and gaze direction through a webcam. It displays blink and long-closure counts, EAR, rolling 60-second PERCLOS, tracking coverage, inference time, and alerts for prolonged eye closure, looking away, or losing face tracking.

Camera frames are processed in the browser and are not uploaded by the app. MediaPipe and its model asset are fetched from external CDNs, so an internet connection is required.

> Research prototype only. DrowseGuard is not a certified safety system, cannot diagnose drowsiness, and must not be used while driving or to make driving decisions.

## Project Structure

```text
Drowse-Guard-main/
|-- .github/
|   `-- workflows/
|       `-- deploy-pages.yml
|-- app.js
|-- index.html
|-- README.md
|-- style.css
|-- vercel.json
`-- Drowse-Guard-main/     # Duplicate project copy; not required to run the root app
```

## Features

- MediaPipe Face Landmarker is the only active face-landmark model.
- Eye Aspect Ratio (EAR) is used for eye-closure detection.
- Blink counts and long eye closures are tracked separately.
- Alerts cover prolonged eye closure, sustained gaze away from the screen, and face loss.
- Rolling 60-second PERCLOS, face-tracking coverage, detection FPS, and inference time are shown.
- Sensitivity and EAR threshold can be adjusted in the app settings.
- Session samples and alerts can be exported as CSV; logs are held in memory and cleared when the page reloads.
- The webcam is requested only after the user starts detection.

Thresholds and alert durations are heuristics, not clinically or experimentally validated values. Session telemetry describes operation, not detection accuracy. Accuracy measurements require independently labeled test sessions.

## Purpose of Each File

- `index.html` - Defines the dashboard, controls, metric displays, alert overlay, and loads the stylesheet and app script.
- `style.css` - Defines the dashboard appearance, layout, and responsive styles.
- `app.js` - Loads MediaPipe, requests webcam access, processes landmarks, calculates EAR/PERCLOS and blink metrics, evaluates gaze and face tracking, displays alerts, and exports session logs.
- `README.md` - Documents the project, features, file roles, setup, and future work.
- `vercel.json` - Configures clean URLs and trailing-slash behavior for static deployment on Vercel.
- `.github/workflows/deploy-pages.yml` - GitHub Actions workflow for deployment.
- `Drowse-Guard-main/` - A duplicate project copy in the workspace. The root `index.html` uses the root `app.js` and `style.css`; this nested copy is not needed for local runs.

## Commands to Run

No package installation or build step is required. Open a terminal in the project root and start a local static server:

```powershell
python -m http.server 8000
```

Open `http://localhost:8000` in a browser, allow camera access, and select **Start Detection**. Use `Ctrl+C` in the terminal to stop the server. A local server is preferred over opening `index.html` directly because browser camera access generally requires `localhost` or HTTPS.

To check the JavaScript syntax with Node.js:

```powershell
node --check app.js
```

## Future Scope

- Calibrate gaze and eye-closure thresholds for each user and camera position, then evaluate whether calibration improves reliability.
- Evaluate blink, gaze-away, and eye-closure detection on consented, labeled data across varied lighting, camera angles, and eyewear.
- Report precision, recall, false-alert rate, missed-event rate, and detection delay instead of treating live telemetry as accuracy.
- Add focused automated tests for landmark processing, alert timing, and reset behavior.
- Consider comparing additional models only when there is suitable labeled data and a clearly defined evaluation protocol.

Any data collection involving people should follow applicable consent and ethics requirements, minimize identifiable data, and document retention and deletion practices. Do not collect data on public roads or by inducing sleep deprivation.
