import argparse
import os
from pathlib import Path
from ultralytics import YOLO

try:
    from prep_shanghaitech import convert_shanghaitech_to_yolo
except ImportError:
    convert_shanghaitech_to_yolo = None


def build_dataset_yaml(target_dir: Path) -> Path:
    yaml_path = target_dir / "crowd_dataset.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found at {yaml_path}. Please prepare the dataset first.")
    return yaml_path


def evaluate_model(model: YOLO, data_yaml: str, batch: int, imgsz: int):
    print("Running validation on the held-out dataset...")
    results = model.val(data=data_yaml, batch=batch, imgsz=imgsz)
    print(results)
    if results and len(results) > 0:
        metrics = getattr(results[0], 'metrics', None)
        if metrics is not None:
            print("Validation metrics:")
            if isinstance(metrics, dict):
                for key, value in metrics.items():
                    try:
                        print(f"  {key}: {float(value):.4f}")
                    except Exception:
                        print(f"  {key}: {value}")
            else:
                print(metrics)
    return results


def main(args):
    target_dir = Path(args.target_dir).expanduser().resolve()
    if args.shanghaitech_dir:
        if convert_shanghaitech_to_yolo is None:
            raise RuntimeError("prep_shanghaitech.py is required for ShanghaiTech conversion.")
        source_dir = Path(args.shanghaitech_dir).expanduser().resolve()
        if not source_dir.exists():
            raise FileNotFoundError(f"ShanghaiTech data root not found: {source_dir}")
        print(f"Converting ShanghaiTech data from {source_dir} to YOLO format in {target_dir}...")
        source_paths = [str(source_dir)]
        target_dir.mkdir(parents=True, exist_ok=True)
        convert_shanghaitech_to_yolo(source_paths, str(target_dir), box_size=args.box_size)

    data_yaml = args.data_yaml
    if data_yaml is None:
        data_yaml_path = build_dataset_yaml(target_dir)
    else:
        data_yaml_path = Path(data_yaml).expanduser().resolve()
        if not data_yaml_path.exists():
            raise FileNotFoundError(f"Data YAML not found: {data_yaml_path}")

    print(f"Using dataset config: {data_yaml_path}")
    print(f"Loading model weights: {args.model}")
    model = YOLO(args.model)

    print("Starting training...")
    training_results = model.train(
        data=str(data_yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(args.project),
        name=args.name,
        device=args.device,
        save=True,
        exist_ok=True,
    )
    print("Training complete.")

    best_weights = Path(args.project) / args.name / "weights" / "best.pt"
    if best_weights.exists():
        print(f"Best weights saved at {best_weights}")
    else:
        print("Warning: best.pt was not found after training.")

    evaluate_model(model, str(data_yaml_path), args.batch, args.imgsz)
    print("Training and evaluation finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a YOLOv8 crowd detection model.")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                        help="Base YOLO weights to fine-tune")
    parser.add_argument("--data-yaml", type=str, default=None,
                        help="Path to YOLO data YAML file")
    parser.add_argument("--shanghaitech-dir", type=str, default=None,
                        help="Root folder of ShanghaiTech dataset for conversion")
    parser.add_argument("--target-dir", type=str, default="./datasets/crowd",
                        help="Target directory for converted dataset")
    parser.add_argument("--project", type=str, default="./runs/train",
                        help="YOLO training project output directory")
    parser.add_argument("--name", type=str, default="crowd_yolov8",
                        help="Experiment name")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Training input image size")
    parser.add_argument("--batch", type=int, default=16,
                        help="Training batch size")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to run on, e.g. cpu or 0 for GPU")
    parser.add_argument("--box-size", type=int, default=24,
                        help="Box size in pixels for YOLO label generation from ShanghaiTech points")
    args = parser.parse_args()
    main(args)
