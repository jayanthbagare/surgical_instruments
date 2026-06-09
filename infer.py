"""
Surgical instrument detector — offline inference on images or live video.

Usage:
    # single image
    python infer.py --source path/to/image.jpg

    # folder of images
    python infer.py --source path/to/folder/

    # live camera (default webcam = 0; change to 1/2 for external cameras)
    python infer.py --source 0

    # video file
    python infer.py --source path/to/video.mp4

    # use a specific trained model
    python infer.py --source 0 --weights runs/surgical/weights/best.pt

Press  q  or  Esc  to quit live/video mode.
"""

import argparse
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


CLASS_NAMES = [
    "Needle Holder", "Dissecting Forceps", "Tissue Forceps",
    "Mosquito Clamp", "Kocher Clamp", "Allis Clamp", "Sponge Forceps",
    "Babcock Clamp", "Right Angle Clamp", "Towel Clamp", "Foerster Clamp",
    "Retractor", "Volkmann Curette", "Bone Curette", "Periosteal Elevator",
    "Osteotome", "Bone Chisel", "Mallet", "Bone Rasp", "Gigli Saw",
    "Wire Cutter", "Suture Scissors", "Mayo Scissors", "Metzenbaum Scissors",
    "Bandage Scissors", "Scalpel",
]

# Assign a distinct BGR colour to each class
rng = np.random.default_rng(42)
PALETTE = rng.integers(80, 220, size=(len(CLASS_NAMES), 3)).tolist()

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source",  default="0",
                   help="Image / folder / video file / camera index (default 0 = webcam)")
    p.add_argument("--weights", default="runs/surgical/weights/best.pt",
                   help="Path to trained YOLOv8 weights (.pt)")
    p.add_argument("--conf",    type=float, default=0.35,
                   help="Confidence threshold")
    p.add_argument("--iou",     type=float, default=0.45,
                   help="NMS IoU threshold")
    p.add_argument("--imgsz",   type=int,   default=640)
    p.add_argument("--save",    action="store_true",
                   help="Save annotated output next to source file")
    return p.parse_args()


# ── drawing helpers ────────────────────────────────────────────────────────────

def draw_box(frame, x1, y1, x2, y2, label, conf, colour):
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
    text    = f"{label} {conf:.0%}"
    (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    by = max(y1 - 4, th + bl)
    cv2.rectangle(frame, (x1, by - th - bl), (x1 + tw + 4, by + bl), colour, -1)
    cv2.putText(frame, text, (x1 + 2, by),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def draw_count_panel(frame, counts: Counter, fps: float | None = None):
    """Overlay a dark panel in the top-right corner listing instrument counts."""
    lines = [f"TOTAL: {sum(counts.values())}"] + \
            [f"  {name}: {n}" for name, n in sorted(counts.items())]
    if fps:
        lines.append(f"FPS: {fps:.1f}")

    padding = 8
    line_h  = 22
    font    = cv2.FONT_HERSHEY_SIMPLEX
    scale   = 0.55
    thick   = 1

    max_w = max(cv2.getTextSize(l, font, scale, thick)[0][0] for l in lines)
    panel_w = max_w + 2 * padding
    panel_h = len(lines) * line_h + 2 * padding

    H, W = frame.shape[:2]
    x0 = W - panel_w - 10
    y0 = 10

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    for i, line in enumerate(lines):
        colour = (0, 255, 180) if i == 0 else (220, 220, 220)
        cy = y0 + padding + (i + 1) * line_h - 4
        cv2.putText(frame, line, (x0 + padding, cy),
                    font, scale, colour, thick, cv2.LINE_AA)


# ── inference ─────────────────────────────────────────────────────────────────

def run_on_frame(model, frame, conf, iou):
    """Return annotated frame + Counter of detected instrument names."""
    results  = model.predict(frame, conf=conf, iou=iou, verbose=False)[0]
    counts   = Counter()

    for box in results.boxes:
        cls_id = int(box.cls[0])
        name   = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
        conf_v = float(box.conf[0])
        colour = PALETTE[cls_id % len(PALETTE)]

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        draw_box(frame, x1, y1, x2, y2, name, conf_v, colour)
        counts[name] += 1

    return frame, counts


def infer_stream(model, source, conf, iou, save):
    cap_src = int(source) if source.isdigit() else source
    cap     = cv2.VideoCapture(cap_src)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source}")
        return

    # Set up writer if saving
    writer = None
    if save and not str(source).isdigit():
        src_path = Path(source)
        out_path = src_path.with_stem(src_path.stem + "_detected")
        fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
        fps_src  = cap.get(cv2.CAP_PROP_FPS) or 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(out_path), fourcc, fps_src, (w, h))
        print(f"Saving to {out_path}")

    t_prev = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame, counts = run_on_frame(model, frame, conf, iou)

        t_now = time.perf_counter()
        fps   = 1.0 / max(t_now - t_prev, 1e-6)
        t_prev = t_now

        draw_count_panel(frame, counts, fps)
        cv2.imshow("Surgical Instrument Detector  [q / Esc = quit]", frame)

        if writer:
            writer.write(frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


def infer_images(model, source, conf, iou, save):
    p = Path(source)
    paths = sorted(p.glob("*")) if p.is_dir() else [p]
    paths = [f for f in paths if f.suffix.lower() in IMG_EXTS]

    if not paths:
        print(f"[ERROR] No images found at {source}")
        return

    for img_path in paths:
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"  [SKIP] Cannot read {img_path}")
            continue

        frame, counts = run_on_frame(model, frame, conf, iou)
        draw_count_panel(frame, counts)

        print(f"{img_path.name}  →  {dict(counts)}")

        if save:
            out = img_path.with_stem(img_path.stem + "_detected")
            cv2.imwrite(str(out), frame)
            print(f"  Saved {out}")

        cv2.imshow(img_path.name, frame)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()
        if key in (ord("q"), 27):
            break


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        print(f"[ERROR] Weights not found: {weights}")
        print("  Train first:  python train.py")
        return

    print(f"Loading model from {weights} …")
    model = YOLO(str(weights))

    source = args.source
    is_live = source.isdigit() or str(source).endswith((".mp4", ".avi", ".mov", ".mkv"))

    if is_live:
        infer_stream(model, source, args.conf, args.iou, args.save)
    else:
        infer_images(model, source, args.conf, args.iou, args.save)


if __name__ == "__main__":
    main()
