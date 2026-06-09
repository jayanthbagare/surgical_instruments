"""
Fine-tune YOLOv8n on the surgical instrument dataset (OSI-26).

Usage:
    python train.py [--epochs 100] [--imgsz 640] [--batch 8] [--model yolov8n.pt]

The best weights are saved to:
    runs/surgical/weights/best.pt

After training, the script exports to CoreML (for iOS/iPad in theatre)
and ONNX (for any offline laptop/PC).
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


DATASET_YAML = Path(__file__).parent / "yolo_dataset" / "dataset.yaml"
RUNS_DIR     = Path(__file__).parent / "runs"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",  default="yolov8n.pt",
                   help="Pretrained backbone. yolov8n.pt is fastest; yolov8s.pt is more accurate.")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz",  type=int, default=640)
    p.add_argument("--batch",  type=int, default=8,
                   help="Reduce to 4 if you hit memory limits on a laptop GPU/MPS.")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--patience", type=int, default=20,
                   help="Early-stop if val/mAP50 doesn't improve for this many epochs.")
    return p.parse_args()


def main():
    args = parse_args()

    model = YOLO(args.model)

    results = model.train(
        data      = str(DATASET_YAML),
        epochs    = args.epochs,
        imgsz     = args.imgsz,
        batch     = args.batch,
        workers   = args.workers,
        patience  = args.patience,
        project   = str(RUNS_DIR),
        name      = "surgical",
        exist_ok  = True,
        device    = "mps",          # Apple Silicon GPU; change to "0" for CUDA, "cpu" otherwise
        amp       = False,          # AMP can be unstable on MPS; disable for safety
        augment   = True,
        # Augmentation to help with OR lighting & partial occlusions
        hsv_h     = 0.015,
        hsv_s     = 0.5,
        hsv_v     = 0.4,
        fliplr    = 0.5,
        flipud    = 0.0,            # instruments aren't upside-down
        mosaic    = 1.0,
        close_mosaic = 10,
        scale     = 0.5,
        translate = 0.1,
    )

    print("\n── Training complete ──")
    print(f"  Best weights : {RUNS_DIR}/surgical/weights/best.pt")
    print(f"  Last weights : {RUNS_DIR}/surgical/weights/last.pt")

    best_weights = RUNS_DIR / "surgical" / "weights" / "best.pt"
    if not best_weights.exists():
        print("[WARN] best.pt not found, skipping export.")
        return

    # ── Export for offline deployment ─────────────────────────────────────────
    trained = YOLO(str(best_weights))

    print("\nExporting ONNX (CPU inference on any offline device)…")
    trained.export(format="onnx", imgsz=args.imgsz, simplify=True, opset=12)

    print("Exporting CoreML (iPad / iPhone in theatre)…")
    trained.export(format="coreml", imgsz=args.imgsz, nms=True)

    print("\nExport done. Files are alongside best.pt in the weights/ folder.")


if __name__ == "__main__":
    main()
