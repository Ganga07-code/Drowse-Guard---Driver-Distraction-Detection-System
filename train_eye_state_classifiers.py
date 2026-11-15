import argparse
import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path

from ultralytics import YOLO


CLASS_FOLDERS = {
    "Closed": "Closed",
    "Drowsy/Microsleep": "Drowsy-Microsleep",
    "Open": "Open",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train experimental YOLOv8n-cls and YOLO11n-cls eye-state classifiers."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).parent / "datasets" / "simuletic-dms",
        help="Extracted Simuletic_DMS_Dataset directory containing images/ and labels/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("trained_models"),
        help="Directory for training runs and exported ONNX models.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument(
        "--validation-sequence",
        default="driver_no_sleep.json",
        help="Sequence held out from training; the remaining sequences are train-only.",
    )
    return parser.parse_args()


def prepare_dataset(source, destination, validation_sequence):
    label_files = sorted((source / "labels").glob("*.json"))
    if not label_files:
        raise FileNotFoundError(f"No annotation JSON files found in {source / 'labels'}")

    sequence_samples = []
    for label_file in label_files:
        with label_file.open(encoding="utf-8") as stream:
            annotation = json.load(stream)
        samples = annotation.get("samples")
        if not isinstance(samples, list):
            raise ValueError(f"{label_file} does not contain a samples list")
        sequence_samples.append((label_file.name, samples))

    if validation_sequence not in {name for name, _ in sequence_samples}:
        raise ValueError(f"Validation sequence not found: {validation_sequence}")

    counts = {"train": Counter(), "val": Counter()}
    for sequence, samples in sequence_samples:
        split = "val" if sequence == validation_sequence else "train"
        for sample in samples:
            image_name = sample.get("image")
            eye_state = sample.get("attributes", {}).get("eye_state")
            if eye_state not in CLASS_FOLDERS:
                raise ValueError(f"Unsupported eye_state {eye_state!r} in {sequence}")
            image = source / "images" / image_name
            if not image.is_file():
                raise FileNotFoundError(f"Image referenced by {sequence} is missing: {image}")
            target_dir = destination / split / CLASS_FOLDERS[eye_state]
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image, target_dir / image.name)
            counts[split][eye_state] += 1

    if len(counts["train"]) != len(CLASS_FOLDERS):
        raise ValueError(f"Training split lacks one or more classes: {counts['train']}")
    if len(counts["val"]) != len(CLASS_FOLDERS):
        raise ValueError(f"Validation split lacks one or more classes: {counts['val']}")

    print(f"Sequence-level validation holdout: {validation_sequence}")
    print(f"Training label counts: {dict(counts['train'])}")
    print(f"Validation label counts: {dict(counts['val'])}")
    return counts


def main():
    args = parse_args()
    if args.epochs < 1 or args.imgsz < 32 or args.batch < 1:
        raise ValueError("epochs and batch must be positive, and imgsz must be at least 32")
    source = args.dataset.resolve()
    if not (source / "images").is_dir() or not (source / "labels").is_dir():
        raise FileNotFoundError(f"Expected images/ and labels/ under {source}")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    models = (
        ("yolov8n-cls.pt", "yolov8n-eye-state"),
        ("yolo11n-cls.pt", "yolo11n-eye-state"),
    )
    with tempfile.TemporaryDirectory(prefix="drowseguard-eye-state-") as temp_dir:
        dataset_dir = Path(temp_dir) / "dataset"
        prepare_dataset(source, dataset_dir, args.validation_sequence)
        for weights, run_name in models:
            model = YOLO(weights)
            results = model.train(
                data=str(dataset_dir),
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=args.batch,
                device="cpu",
                workers=0,
                project=str(output),
                name=run_name,
                exist_ok=False,
                pretrained=True,
                seed=42,
                deterministic=True,
                plots=True,
                verbose=True,
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
            print(f"{run_name} ONNX export: {exported}")


if __name__ == "__main__":
    main()
