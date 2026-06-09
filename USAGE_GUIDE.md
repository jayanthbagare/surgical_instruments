# Surgical Instrument Detector — Usage Guide

Detects and counts 26 surgical instruments in images or live video.  
Runs **fully offline** — no internet connection required in the operating theatre.

---

## What's in this folder

| File | Purpose |
|---|---|
| `prepare_dataset.py` | Converts XML annotations → YOLO format (run once) |
| `train.py` | Fine-tunes YOLOv8n on the converted dataset |
| `infer.py` | Live camera / video / image inference with count overlay |
| `yolo_dataset/` | Converted dataset (created by `prepare_dataset.py`) |
| `runs/surgical/weights/best.pt` | Trained model weights (created by `train.py`) |
| `runs/surgical/weights/best.onnx` | ONNX export — runs on any laptop/PC offline |
| `runs/surgical/weights/best.mlpackage` | CoreML export — runs on iPad/iPhone |

---

## Requirements

All dependencies are in the Anaconda environment on this machine.  
**Always use `/Users/I028960/anaconda3/bin/python`**, not the system Python.

```bash
# One-time check
/Users/I028960/anaconda3/bin/python -c "import ultralytics, cv2, torch; print('OK')"
```

---

## Step 1 — Prepare the dataset (run once)

Converts the Pascal VOC XML annotations to YOLO format and writes `yolo_dataset/`.

```bash
cd /Users/I028960/Desktop/Kushal
/Users/I028960/anaconda3/bin/python prepare_dataset.py
```

Expected output:
```
Converted 411 samples → /Users/I028960/Desktop/Kushal/yolo_dataset
Wrote /Users/I028960/Desktop/Kushal/yolo_dataset/dataset.yaml
```

This step is already done — only re-run if you add new images/annotations.

---

## Step 2 — Train the model

```bash
cd /Users/I028960/Desktop/Kushal
/Users/I028960/anaconda3/bin/python train.py
```

### Recommended settings by hardware

| Scenario | Command |
|---|---|
| MacBook (Apple Silicon) — default | `python train.py` |
| Faster training, slightly larger model | `python train.py --model yolov8s.pt` |
| Low memory (crashes) | `python train.py --batch 4` |
| Quick test run | `python train.py --epochs 5 --batch 4` |
| Full high-accuracy run | `python train.py --model yolov8s.pt --epochs 200 --batch 8` |

### Key arguments

| Argument | Default | Meaning |
|---|---|---|
| `--model` | `yolov8n.pt` | Backbone: `n`=nano (fastest), `s`=small, `m`=medium |
| `--epochs` | `100` | Training epochs (more = better, up to a point) |
| `--batch` | `8` | Images per step — reduce if you get memory errors |
| `--imgsz` | `640` | Input resolution in pixels |
| `--patience` | `20` | Early-stop if no improvement for N epochs |

### Training output

```
runs/surgical/
  weights/
    best.pt          ← use this for inference
    last.pt          ← last checkpoint
    best.onnx        ← for offline Windows/Linux laptops
    best.mlpackage   ← for iPad/iPhone in theatre
  results.png        ← loss + mAP curves
  confusion_matrix.png
  val_batch0_pred.jpg ← visual check of predictions
```

Training on Apple Silicon (M1/M2/M3) typically takes:
- ~15–25 min for 100 epochs with `yolov8n`
- ~35–50 min for 100 epochs with `yolov8s`

---

## Step 3 — Run inference

### Live camera (operating theatre)

```bash
/Users/I028960/anaconda3/bin/python infer.py --source 0
```

- `0` = built-in webcam. Change to `1`, `2` etc. for an external USB camera.
- Press **q** or **Esc** to quit.

### Single image

```bash
/Users/I028960/anaconda3/bin/python infer.py --source path/to/image.jpg
```

Press any key to advance, **q** to quit.

### Folder of images

```bash
/Users/I028960/anaconda3/bin/python infer.py --source path/to/folder/
```

### Video file

```bash
/Users/I028960/anaconda3/bin/python infer.py --source path/to/video.mp4
```

### Save annotated output

Add `--save` to write annotated files alongside the originals:

```bash
/Users/I028960/anaconda3/bin/python infer.py --source video.mp4 --save
# → saves video_detected.mp4
```

### Use a specific model

```bash
/Users/I028960/anaconda3/bin/python infer.py \
  --source 0 \
  --weights runs/surgical/weights/best.pt
```

### Tune detection sensitivity

| Argument | Default | Effect |
|---|---|---|
| `--conf` | `0.35` | Raise to reduce false positives; lower to catch more instruments |
| `--iou` | `0.45` | NMS overlap threshold — lower if overlapping tools merge |

```bash
# Stricter — fewer but more confident detections
/Users/I028960/anaconda3/bin/python infer.py --source 0 --conf 0.5

# More permissive — catches partially hidden instruments
/Users/I028960/anaconda3/bin/python infer.py --source 0 --conf 0.25
```

---

## What you see on screen

```
┌─────────────────────────────────┐
│                     TOTAL: 6    │
│                       Scalpel: 1│
│                Needle Holder: 2 │
│          Dissecting Forceps: 1  │
│              Mayo Scissors: 2   │
│                     FPS: 18.3   │
│                                 │
│  [bounding boxes on each tool]  │
└─────────────────────────────────┘
```

Each instrument gets a coloured bounding box with its name and confidence score.  
The count panel (top-right) shows every detected class and the total instrument count.

---

## Deploying on a different offline device

### Windows / Linux laptop (no internet needed)

Copy `best.onnx` to the target machine and run inference with ONNX Runtime:

```bash
pip install onnxruntime opencv-python ultralytics
python infer.py --weights runs/surgical/weights/best.onnx --source 0
```

### iPad / iPhone (CoreML)

`best.mlpackage` (5.9 MB) can be integrated into an iOS/iPadOS app using the  
`Vision` framework or `CoreML` API. It includes NMS so no post-processing is needed.

---

## Instrument classes (1–26)

| ID | Name | ID | Name |
|---|---|---|---|
| 1 | Needle Holder | 14 | Bone Curette |
| 2 | Dissecting Forceps | 15 | Periosteal Elevator |
| 3 | Tissue Forceps | 16 | Osteotome |
| 4 | Mosquito Clamp | 17 | Bone Chisel |
| 5 | Kocher Clamp | 18 | Mallet |
| 6 | Allis Clamp | 19 | Bone Rasp |
| 7 | Sponge Forceps | 20 | Gigli Saw |
| 8 | Babcock Clamp | 21 | Wire Cutter |
| 9 | Right Angle Clamp | 22 | Suture Scissors |
| 10 | Towel Clamp | 23 | Mayo Scissors |
| 11 | Foerster Clamp | 24 | Metzenbaum Scissors |
| 12 | Retractor | 25 | Bandage Scissors |
| 13 | Volkmann Curette | 26 | Scalpel |

---

## Troubleshooting

**"Weights not found" error**
→ Run `train.py` first to generate `runs/surgical/weights/best.pt`.

**Camera not opening**
→ Try `--source 1` or `--source 2`. On macOS, grant Terminal camera permission in  
  System Settings → Privacy & Security → Camera.

**Out of memory during training**
→ Add `--batch 4` or `--batch 2`.

**Low accuracy / missing detections**
→ Train longer: `--epochs 200`. Or use a bigger backbone: `--model yolov8s.pt`.  
→ Lower the confidence threshold at inference time: `--conf 0.25`.

**Slow frame rate on camera**
→ `yolov8n` (nano) is already the fastest variant. Reduce `--imgsz 320` for more speed  
  at the cost of some accuracy.
