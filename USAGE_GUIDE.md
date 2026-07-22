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

Dependencies are managed with [`uv`](https://docs.astral.sh/uv/) and pinned in
`requirements.txt` (web app + ML) and `requirements-dev.txt` (tests).

```bash
# One-time setup — creates .venv and installs everything
uv venv
uv pip install -r requirements.txt
uv pip install -r requirements-dev.txt   # only needed to run tests

# Sanity check (no imports errors)
uv run python -c "import ultralytics, cv2, torch; print('OK')"
```

> **Why `opencv-python-headless`?** The server uses the headless OpenCV build,
> which has no system-library dependencies (`libxcb`, `libGL`, …). The GUI
> variant (`opencv-python`) — pulled in transitively by `ultralytics` — fails to
> import on headless servers with `ImportError: libxcb.so.1`. `requirements.txt`
> lists `opencv-python-headless` *after* `ultralytics` so it wins. The CLI
> (`infer.py`) detects the absence of a display and skips the preview window,
> so batch processing and `--save` still work.

---

## Step 1 — Prepare the dataset (run once)

Converts the Pascal VOC XML annotations to YOLO format and writes `yolo_dataset/`.

```bash
uv run python prepare_dataset.py
```

Expected output:
```
Converted 411 samples → yolo_dataset
Wrote yolo_dataset/dataset.yaml
```

This step is already done — only re-run if you add new images/annotations.

---

## Step 2 — Train the model

```bash
uv run python train.py
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
uv run python infer.py --source 0
```

- `0` = built-in webcam. Change to `1`, `2` etc. for an external USB camera.
- Press **q** or **Esc** to quit.

### Single image

```bash
uv run python infer.py --source path/to/image.jpg
```

Press any key to advance, **q** to quit.

### Folder of images

```bash
uv run python infer.py --source path/to/folder/
```

### Video file

```bash
uv run python infer.py --source path/to/video.mp4
```

### Save annotated output

Add `--save` to write annotated files alongside the originals:

```bash
uv run python infer.py --source video.mp4 --save
# → saves video_detected.mp4
```

### Use a specific model

```bash
uv run python infer.py \
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
uv run python infer.py --source 0 --conf 0.5

# More permissive — catches partially hidden instruments
uv run python infer.py --source 0 --conf 0.25
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

## Step 4 — Web frontend (optional)

A small browser UI for uploading an image or video and viewing the annotated
output with bounding boxes + counts. Built with FastAPI (backend) and vanilla
HTML/CSS/JS (frontend — no Node build step). It **reuses the existing functions
in `infer.py`** (`run_on_frame`, `draw_count_panel`, `CLASS_NAMES`, `PALETTE`).

### Install (one-time)

```bash
uv pip install -r requirements.txt
```

### Run

From the project root:

```bash
uv run python -m uvicorn webapp.main:app --port 8000
```

Open http://localhost:8000 in any browser.

- **Image tab** — drop / browse an image → click *Detect instruments* → original
  and annotated images are shown side by side with a counts panel.
- **Video tab** — drop / browse a video → click *Process video* → an annotated
  MP4 (playable + downloadable) and per-class counts are shown.

> Note: tab visibility is driven by the `.panel` / `.panel.active` CSS rules.
> Do **not** add a `hidden` class to a `<section class="panel">` —
> `.hidden { display:none !important }` overrides `.panel.active` and the tab
> will appear blank. The tab switcher in `app.js` toggles both `active` and
> `hidden` defensively.

### Weights

The app auto-selects `runs/surgical/weights/best.pt` if present, otherwise
falls back to the shipped `pre-trained_weights/best.pt`. Override with:

```bash
WEIGHTS=/path/to/best.pt uv run python -m uvicorn webapp.main:app --port 8000
```

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET  | `/api/health` | Model status + class list |
| POST | `/api/infer/image?conf=&iou=` | multipart `file` → JSON `{image, counts, total}` |
| POST | `/api/infer/video?conf=&iou=` | multipart `file` → annotated `video/mp4` (counts in `X-Counts` header) |

---

## Troubleshooting

**`ImportError: libxcb.so.1: cannot open shared object file` (server won't start)**
→ The GUI `opencv-python` build (pulled in by `ultralytics`) needs system
  libraries that headless servers don't have. `requirements.txt` lists
  `opencv-python-headless` *after* `ultralytics` so it overrides the GUI build.
  Re-install in order: `uv pip install -r requirements.txt`. Confirm with
  `uv run python -c "import cv2"`.

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

---

## Tests

```bash
uv pip install -r requirements-dev.txt
uv run pytest
```

Covers the FastAPI endpoints (`/api/health`, image inference) and the
`infer.py` drawing/count helpers. The model is stubbed, so tests run fully
offline without weights or a GPU.
