import argparse
import json
import tempfile
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO


GAZE_ANNOTATION_SUFFIX = "_rgb_ann_gaze.json"
FACE_VIDEO_SUFFIX = "_rgb_face.mp4"
DEFAULT_DATASET = Path(__file__).parent / "dmd"
DEFAULT_OUTPUT = Path(__file__).parent / "trained_models"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train experimental YOLOv8n-cls and YOLO11n-cls gaze-zone classifiers."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="DMD dataset directory containing one *_rgb_ann_gaze.json and matching *_rgb_face.mp4.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument(
        "--validation-gap",
        type=int,
        default=30,
        help="Unassigned video frames around each train/validation boundary.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--patience", type=int, default=15)
    return parser.parse_args()


def find_recording(dataset):
    annotations = sorted(dataset.rglob(f"*{GAZE_ANNOTATION_SUFFIX}"))
    if len(annotations) != 1:
        raise ValueError(
            f"Expected exactly one gaze annotation in {dataset}, found {len(annotations)}. "
            "Use a single recording; multi-recording, subject-independent splitting is not "
            "implemented."
        )

    annotation = annotations[0]
    video_name = annotation.name.removesuffix(GAZE_ANNOTATION_SUFFIX) + FACE_VIDEO_SUFFIX
    video = annotation.with_name(video_name)
    if not video.is_file():
        raise FileNotFoundError(f"Matching face video not found: {video}")
    return annotation, video


def parse_gaze_intervals(annotation_path, validation_fraction, validation_gap, frame_stride):
    with annotation_path.open(encoding="utf-8") as stream:
        annotation = json.load(stream)
    openlabel = annotation.get("openlabel")
    if not isinstance(openlabel, dict) or not isinstance(openlabel.get("actions"), dict):
        raise ValueError(f"{annotation_path} does not contain OpenLABEL actions")

    intervals = []
    classes = set()
    occupied_frames = set()
    for action in openlabel["actions"].values():
        action_type = action.get("type", "")
        if not action_type.startswith("gaze_zone/"):
            continue
        class_name = action_type.split("/", 1)[1]
        if class_name == "not_valid":
            continue
        classes.add(class_name)
        action_intervals = action.get("frame_intervals")
        if not isinstance(action_intervals, list) or not action_intervals:
            raise ValueError(f"Gaze action {action_type!r} has no frame intervals")

        for interval in action_intervals:
            start = interval.get("frame_start")
            end = interval.get("frame_end")
            if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end < start:
                raise ValueError(f"Invalid frame interval in {annotation_path}: {interval}")
            if any(frame in occupied_frames for frame in range(start, end + 1)):
                raise ValueError(f"Overlapping gaze-zone intervals at {action_type!r}")
            occupied_frames.update(range(start, end + 1))
            split_at = start + int((end - start + 1) * (1 - validation_fraction))
            train_end = split_at - validation_gap // 2
            val_start = split_at + (validation_gap - validation_gap // 2)
            train_frames = list(range(start, train_end + 1, frame_stride))
            val_frames = list(range(val_start, end + 1, frame_stride))
            if not train_frames or not val_frames:
                raise ValueError(
                    f"Interval {action_type!r} ({start}-{end}) is too short for the "
                    "requested validation fraction, gap, and frame stride."
                )
            intervals.append((class_name, train_frames, val_frames))

    if len(classes) < 2:
        raise ValueError(f"Expected at least two valid gaze-zone classes, found {sorted(classes)}")
    return sorted(classes), intervals


def prepare_dataset(annotation_path, video_path, destination, args):
    classes, intervals = parse_gaze_intervals(
        annotation_path,
        args.validation_fraction,
        args.validation_gap,
        args.frame_stride,
    )
    assignments = {}
    for class_name, train_frames, val_frames in intervals:
        for split, frames in (("train", train_frames), ("val", val_frames)):
            for frame_index in frames:
                if frame_index in assignments:
                    raise ValueError(f"Frame {frame_index} is assigned more than once")
                assignments[frame_index] = (split, class_name)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise OSError(f"Could not open face video: {video_path}")
    reported_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if reported_frames and max(assignments) >= reported_frames:
        capture.release()
        raise ValueError(
            f"Gaze annotations reference frame {max(assignments)}, but the video has "
            f"{reported_frames} frames"
        )

    counts = {"train": Counter(), "val": Counter()}
    try:
        for frame_index in range(max(assignments) + 1):
            success, frame = capture.read()
            if not success:
                if frame_index in assignments:
                    raise OSError(f"Could not decode labeled frame {frame_index} from {video_path}")
                break
            assignment = assignments.get(frame_index)
            if assignment is None:
                continue

            split, class_name = assignment
            height, width = frame.shape[:2]
            side = min(height, width)
            x = (width - side) // 2
            y = (height - side) // 2
            square_frame = frame[y : y + side, x : x + side]
            target_dir = destination / split / class_name
            target_dir.mkdir(parents=True, exist_ok=True)
            image_path = target_dir / f"frame_{frame_index:06d}.jpg"
            if not cv2.imwrite(str(image_path), square_frame):
                raise OSError(f"Could not write prepared frame: {image_path}")
            counts[split][class_name] += 1
    finally:
        capture.release()

    for split in ("train", "val"):
        missing = sorted(set(classes) - set(counts[split]))
        if missing:
            raise ValueError(f"{split} split lacks gaze-zone classes: {missing}")

    print(f"Gaze-zone classes (folder/model order): {', '.join(classes)}")
    print(f"Training frame counts: {dict(sorted(counts['train'].items()))}")
    print(f"Validation frame counts: {dict(sorted(counts['val'].items()))}")
    print(
        "Validation is a temporal holdout from this same recording with the configured "
        f"{args.validation_gap}-frame gap; it is not an independent-subject evaluation."
    )
    return classes


def train_models(dataset_dir, output, classes, args):
    models = (
        ("yolov8n-cls.pt", "yolov8n-gaze-zones"),
        ("yolo11n-cls.pt", "yolo11n-gaze-zones"),
    )
    for weights, run_name in models:
        model = YOLO(weights)
        results = model.train(
            data=str(dataset_dir),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=0,
            project=str(output),
            name=run_name,
            exist_ok=False,
            pretrained=True,
            seed=42,
            deterministic=True,
            plots=True,
            verbose=True,
            patience=args.patience,
            fliplr=0.0,
        )
        best_weights = Path(results.save_dir) / "weights" / "best.pt"
        if not best_weights.is_file():
            raise FileNotFoundError(f"Training did not produce best.pt in {results.save_dir}")
        exported = YOLO(str(best_weights)).export(
            format="onnx",
            imgsz=args.imgsz,
            batch=1,
            dynamic=False,
            half=False,
        )
        print(f"{run_name} best weights: {best_weights}")
        print(f"{run_name} ONNX export: {exported}")
        print(f"{run_name} class order: {', '.join(classes)}")


def main():
    args = parse_args()
    if (
        args.epochs < 1
        or args.imgsz < 32
        or args.batch < 1
        or args.frame_stride < 1
        or args.validation_gap < 0
        or args.patience < 1
        or not 0 < args.validation_fraction < 1
    ):
        raise ValueError("Invalid training or dataset-splitting options")

    dataset = args.dataset.resolve()
    if not dataset.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset}")
    annotation_path, video_path = find_recording(dataset)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="drowseguard-dmd-gaze-") as temp_dir:
        dataset_dir = Path(temp_dir) / "dataset"
        classes = prepare_dataset(annotation_path, video_path, dataset_dir, args)
        train_models(dataset_dir, output, classes, args)


if __name__ == "__main__":
    main()
