<div align="center">

# 😴 DrowseGuard

### A browser-based drowsiness-monitoring research prototype

Monitor eye closure and gaze cues with your webcam, with live session metrics and alerts.

</div>

🌐 **Live demo:** [Open DrowseGuard](https://drowse-guard-app.vercel.app/)

> [!WARNING]
> **Research prototype only.** DrowseGuard is not a certified safety system and cannot diagnose drowsiness. **Never use it while driving or to make driving decisions.** Its thresholds and alert timings are heuristics, not clinically or experimentally validated values.

## ✨ What it does

DrowseGuard uses **MediaPipe Face Landmarker** in your browser to analyze facial landmarks and display:

- 👁️ Eye Aspect Ratio (EAR), blink counts, and long eye closures
- 📊 Rolling 60-second PERCLOS and face-tracking coverage
- ⚡ Detection FPS and inference time
- 🚨 Alerts for prolonged eye closure, sustained gaze away, and loss of face tracking, with an animated emoji and one short instruction
- 🧭 Head and feature alignment checks using facial landmarks, with a 70% visibility threshold
- 🔔 Distinct alert tones and spoken prompts such as “Please open your eyes” and “Please look straight ahead”
- 📁 CSV export of session samples and alerts

Voice prompts are enabled by default. Voice alerts, their volume, sensitivity, and the EAR threshold can be adjusted in the app settings. The alert dialog keeps the animated emoji and a single clear instruction; the test-preview buttons and extra alert telemetry text have been removed. Session logs are held in memory and cleared when the page reloads.

Detection is scheduled against new webcam frames where the browser supports video-frame callbacks, and the sparkline redraw is limited to 10 updates per second to reduce unnecessary work.

## 🔒 Privacy & requirements

- Camera access is requested only after you start detection.
- Camera frames are processed in the browser and are not uploaded by the app.
- MediaPipe and its model asset are loaded from external CDNs, so an internet connection is required.
- Session telemetry describes app operation, not detection accuracy. Accuracy requires independently labeled test sessions.

## 🚀 Run locally

No package installation or build step is required. From the project root, start a local static server:

```powershell
python -m http.server 8000
```

Then open **http://localhost:8000**, allow camera access, and select **Start Detection**. A local server is preferred to opening `index.html` directly because camera access generally requires `localhost` or HTTPS. Stop the server with `Ctrl+C`.

Check JavaScript syntax with Node.js:

```powershell
node --check app.js
```

## 🗂️ Project structure

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

### Main files

| File | Purpose |
| --- | --- |
| `index.html` | Dashboard, controls, metric displays, and alert overlay |
| `style.css` | Dashboard layout, appearance, and responsive styles |
| `app.js` | MediaPipe loading, webcam handling, landmark processing, metrics, alerts, and CSV export |
| `vercel.json` | Clean URL and trailing-slash configuration for static deployment on Vercel |
| `.github/workflows/deploy-pages.yml` | GitHub Actions workflow for deployment |

The root `index.html` loads the root `app.js` and `style.css`. The nested `Drowse-Guard-main/` directory is a duplicate project copy and is not needed to run the root app.

## 🔭 Future work

- Calibrate gaze and eye-closure thresholds for individual users and camera positions, then evaluate whether calibration improves reliability.
- Evaluate detection on consented, labeled data across varied lighting, camera angles, and eyewear.
- Report precision, recall, false-alert rate, missed-event rate, and detection delay rather than treating live telemetry as accuracy.
- Add focused automated tests for landmark processing, alert timing, and reset behavior.
- Compare additional models only when suitable labeled data and a clearly defined evaluation protocol are available.

Any data collection involving people should follow applicable consent and ethics requirements, minimize identifiable data, and document retention and deletion practices. Do not collect data on public roads or by inducing sleep deprivation.
