# 😴 DrowseGuard

**A browser-based driver-monitoring research prototype** for exploring eye closure, blinks, and frontal-face alignment.

> [!WARNING]
> Research use only—not a certified safety system or medical device. Never use DrowseGuard while driving or to make driving decisions. Its measurements and alerts are experimental.

## ✨ What you can test

- **MediaPipe:** eye aspect ratio (EAR), blink transitions, long closures, mouth opening, and face alignment.
- **YOLO ONNX:** run an object detector or display a full-frame classifier prediction.
- **Live telemetry:** inference time, pipeline FPS, face-tracking coverage, and session CSV logs.
- **Private by design:** camera frames and locally selected ONNX files are processed in the browser, not uploaded by the app.

## 🚀 Start the app

From the project root, run:

```powershell
python -m http.server 8000
```

Open [http://localhost:8000](http://localhost:8000), allow camera access, choose a mode, then select **Start Detection**. Use `Ctrl+C` in the server terminal to stop it. A local server is recommended instead of opening `index.html` directly.

MediaPipe assets and ONNX Runtime Web are loaded from CDNs, so an internet connection is required. No package install or build step is needed.

## 🧠 Choose a model

| Mode | What it does | Alert source |
| --- | --- | --- |
| MediaPipe | Tracks face landmarks and eye/mouth metrics | EAR, face alignment, and mouth measurements |
| YOLO frame classification + MediaPipe | Displays a YOLO class prediction while MediaPipe measures eyes and face alignment | MediaPipe only; the YOLO class does not trigger alerts |
| YOLO object detection | Detects the classes supported by the selected model | Only configured detector class IDs |

### 🏷️ Configure a classifier

1. Select **YOLOv8** or **YOLO11** and choose **Frame classification + MediaPipe landmarks**.
2. Load a compatible batch-1, float32, RGB, 224×224 ONNX classifier.
3. Enter every class label in the model's exact output order, separated by commas.
4. Leave **Alert class IDs** blank and keep the confidence threshold at its default; both are for object detection.

For the included eye-state classifiers, the label order is `Closed,Drowsy-Microsleep,Open`. These are small, experimental full-frame models—not eye detectors. In classifier mode, MediaPipe EAR drives eye-closure and blink measurements. The app estimates frontal-face alignment from landmark symmetry; below 70% for about 1.2 seconds triggers an alert. This score is not a literal percentage of visible face pixels.

The DMD gaze-zone classifiers use this class order:

```text
center_mirror,front,front_right,infotainment,left,left_mirror,right,right_mirror,steering_wheel
```

They display an experimental YOLO prediction but do not trigger gaze alerts. Gaze alerts come from MediaPipe face alignment.

### 🎯 Configure an object detector

Use a batch-1, float32, RGB, 640×640 ONNX model with raw YOLO detection output or end-to-end NMS output. Enter its labels in model order and the **zero-based** IDs that should trigger alerts. The confidence threshold applies only to detections. The standard COCO `yolov8n` and `yolo11n` models are not trained to detect eye closure or drowsiness.

For a compatible Ultralytics export:

```powershell
yolo export model=best.pt format=onnx imgsz=640 batch=1 dynamic=False half=False
```

A successful export does not establish accuracy; validate with independently labeled data.

## 🏋️ Train the included experiments

Run commands from the project root with Ultralytics installed in `.venv`.

### 👁️ Eye-state classifiers

```powershell
.\.venv\Scripts\python.exe .\train_eye_state_classifiers.py `
  --output .\trained_models `
  --epochs 30
```

Outputs:

- `trained_models\yolov8n-eye-state\weights\best.onnx`
- `trained_models\yolo11n-eye-state\weights\best.onnx`

The sample is the [Simuletic Driver Drowsiness Detection Dataset (Synthetic)](https://www.kaggle.com/datasets/simuletic/driver-drowsiness-detection-monitoring-dataset), licensed **CC BY-NC-SA 4.0**. The project copy contains 168 JPG frames and six annotation JSONs under `datasets\simuletic-dms`; see its `ATTRIBUTION.txt`. The recorded validation was only 75.86% top-1 on 29 frames, with one `Closed` example. This tiny, imbalanced split is not evidence of real-world accuracy. Review the dataset and Ultralytics terms before distributing data or model weights.

### 🧭 DMD gaze-zone classifiers

```powershell
.\.venv\Scripts\python.exe .\train_dmd_gaze_classifiers.py `
  --dataset .\dmd `
  --output .\trained_models `
  --epochs 50
```

Outputs are saved under `trained_models\yolov8n-gaze-zones` and `trained_models\yolo11n-gaze-zones`. This experimental training path currently expects one annotated recording; its temporal validation is not an independent-driver test. The script reads the original `dmd` files in place. Check dataset and Ultralytics terms before redistributing data or weights.

## ⚡ Performance and privacy

YOLO inference uses the ONNX Runtime WebAssembly worker proxy, and the hybrid classifier path overlaps YOLO inference with MediaPipe landmark processing. This reduces main-thread work, but achievable FPS still depends on the browser, device, model, and camera. Session logs stay in memory until reload and can be exported as CSV. Telemetry measures app operation—not model accuracy.

## 🗂️ Project files

| File/folder | Purpose |
| --- | --- |
| `index.html`, `style.css`, `app.js` | Dashboard, styling, detection, metrics, and alerts |
| `train_eye_state_classifiers.py` | Eye-state training and ONNX export |
| `train_dmd_gaze_classifiers.py` | DMD gaze-zone training and ONNX export |
| `datasets\simuletic-dms` | Licensed eye-state sample and attribution |
| `dmd` | DMD gaze training data |

Check JavaScript syntax with:

```powershell
node --check app.js
```
