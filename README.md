<div align="center">

# 😴 DrowseGuard

### A browser-based drowsiness-monitoring research prototype

Monitor eye closure and gaze cues with your webcam, with live session metrics and alerts.

</div>

🌐 **Live demo:** [Open DrowseGuard](https://drowse-guard-app.vercel.app/)

> [!WARNING]
> **Research prototype only.** DrowseGuard is not a certified safety system and cannot diagnose drowsiness. **Never use it while driving or to make driving decisions.** Its thresholds and alert timings are heuristics, not clinically or experimentally validated values.

## ✨ What it does

DrowseGuard provides MediaPipe landmark tracking and optional YOLO ONNX inference. YOLO can run an object detector or one of the experimental full-frame eye-state classifiers; MediaPipe is not loaded in YOLO modes.

- 👁️ Eye Aspect Ratio (EAR), blink counts, and long eye closures
- 📊 Rolling 60-second PERCLOS and face-tracking coverage
- ⚡ Pipeline FPS and MediaPipe inference time
- 🚨 A clean alert dialog for prolonged eye closure, sustained gaze away, loss of face tracking, and sustained mouth opening
- 🧭 Head and feature alignment checks using facial landmarks, with a 70% visibility threshold
- 🔔 Distinct alert tones and spoken prompts such as “Please open your eyes” and “Please look straight ahead”
- 📁 CSV export of session samples and alerts

Voice prompts are enabled by default. Voice alerts, their volume, sensitivity, and the EAR threshold can be adjusted in the app settings. Session logs are held in memory and cleared when the page reloads.

To use YOLO inference, choose YOLOv8 or YOLO11, select the model task, and load a compatible ONNX file. Models run locally in the browser with ONNX Runtime Web; the selected file is not uploaded and is cleared when the page reloads. For object detection, enter labels and alert class IDs that match the model. The general-purpose `yolov8n` and `yolo11n` COCO weights are not trained to detect drowsiness.

The app also supports a **batch-1, float32, RGB, 224×224 classification model**. The experimental classifiers trained by this project distinguish `Closed`, `Drowsy-Microsleep`, and `Open` using the entire frame; they do not detect eye regions or gaze-away. In the model controls, choose **Eye-state classification**, load the exported classifier, enter labels in order as `Closed,Drowsy-Microsleep,Open`, and enter alert class IDs `0,1`. Alerts are triggered after the configured eye-state remains predicted for the selected sensitivity duration. Blink counts are approximate open/closed transitions, not independently validated blink detections. No classifier-based gaze, yawning, or face-presence alert is supported.

The existing object-detection adapter expects a **batch-1, float32, RGB, 640×640 model** with raw YOLO detection output or end-to-end NMS output. It does not decode custom face-keypoint heads; the existing MediaPipe model supplies those landmarks. For example, export a compatible detector from a Python environment with Ultralytics installed:

```powershell
yolo export model=best.pt format=onnx imgsz=640 batch=1 dynamic=False half=False
```

Use the resulting `.onnx` file in the corresponding model picker. Check that the exported model's classes and order match the labels and drowsy class IDs configured in the app. A successful export alone does not validate drowsiness accuracy; evaluate models against independently labeled data before drawing conclusions.

### Experimental eye-state classifier training

The included training script creates a sequence-level train/validation split (holding out `driver_no_sleep.json` by default), trains both YOLOv8n-cls and YOLO11n-cls on CPU, and exports ONNX files. Install Ultralytics in the Python environment, then run from the project root:

```powershell
.\.venv\Scripts\python.exe .\train_eye_state_classifiers.py `
  --output .\trained_models `
  --epochs 30
```

The dataset is included under `datasets\simuletic-dms`; pass `--dataset` to use another extracted copy. The exported models are `trained_models\yolov8n-eye-state\weights\best.onnx` and `trained_models\yolo11n-eye-state\weights\best.onnx`. Choose each file in its matching model panel and configure it for eye-state classification. In this run, both models recorded 75.86% top-1 on 29 held-out frames, of which only one was labeled `Closed`; this imbalanced, tiny validation set cannot establish accuracy. Treat this only as a software integration experiment, not evidence of accuracy or readiness for real-world use. The labels include only on-road gaze while eyes are open and do not support gaze-away training.

The sample is the **[Simuletic Driver Drowsiness Detection Dataset (Synthetic)](https://www.kaggle.com/datasets/simuletic/driver-drowsiness-detection-monitoring-dataset)** and is labeled **CC BY-NC-SA 4.0**. This project copy includes the 168 JPG frames and six annotation JSONs; it omits the source videos. See `datasets\simuletic-dms\ATTRIBUTION.txt`. Attribute Simuletic and retain the non-commercial/share-alike terms for dataset adaptations and model derivatives. The training script also initializes from Ultralytics pretrained classifier weights; review the applicable upstream terms before redistribution. Do not treat the resulting weights as cleared for commercial use.

Detection is scheduled against new webcam frames where the browser supports video-frame callbacks, and the sparkline redraw is limited to 10 updates per second to reduce unnecessary work.

## 🔒 Privacy & requirements

- Camera access is requested only after you start detection.
- Camera frames are processed in the browser and are not uploaded by the app.
- MediaPipe, ONNX Runtime Web, and their runtime/model assets require an internet connection. YOLO ONNX weights are selected locally and processed in the browser.
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
|-- datasets/
|   `-- simuletic-dms/   # Licensed sample images and eye-state annotations
|-- train_eye_state_classifiers.py
|-- vercel.json
`-- Drowse-Guard-main/     # Duplicate project copy; not required to run the root app
```

### Main files

| File | Purpose |
| --- | --- |
| `index.html` | Dashboard, controls, metric displays, and alert overlay |
| `style.css` | Dashboard layout, appearance, and responsive styles |
| `app.js` | MediaPipe loading, webcam handling, landmark processing, metrics, alerts, and CSV export |
| `train_eye_state_classifiers.py` | Sequence-split training and ONNX export for experimental eye-state classifiers |
| `vercel.json` | Clean URL and trailing-slash configuration for static deployment on Vercel |
| `.github/workflows/deploy-pages.yml` | GitHub Actions workflow for deployment |

The root `index.html` loads the root `app.js` and `style.css`. The nested `Drowse-Guard-main/` directory is a duplicate project copy and is not needed to run the root app.

## 🔭 Future work

- Calibrate gaze and eye-closure thresholds for individual users and camera positions, then evaluate whether calibration improves reliability.
- Evaluate detection on consented, labeled data across varied lighting, camera angles, and eyewear.
- Report precision, recall, false-alert rate, missed-event rate, and detection delay rather than treating live telemetry as accuracy.
- Add focused automated tests for landmark processing, alert timing, and reset behavior.
- Evaluate compatible YOLO ONNX models with a documented, repeatable protocol and suitable labeled data; add model-specific keypoint decoding only when a supported facial-landmark model is available.
Any data collection involving people should follow applicable consent and ethics requirements, minimize identifiable data, and document retention and deletion practices. Do not collect data on public roads or by inducing sleep deprivation.
