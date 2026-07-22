"""
FastAPI backend for the Surgical Instrument Detector.

Reuses the inference + drawing functions already implemented in ``infer.py``
(``run_on_frame``, ``draw_count_panel``, ``CLASS_NAMES``, ``PALETTE``) so the
detection logic stays in one place.

Endpoints
---------
GET  /api/health            -> { status, weights, classes }
POST /api/infer/image       -> multipart "file" -> JSON {image(dataURI), counts, total}
POST /api/infer/video       -> multipart "file" -> annotated mp4 (X-Counts header)
GET  /                      -> serves the frontend (webapp/static)

Run (from repo root):
    uv run python -m uvicorn webapp.main:app --reload --port 8000
"""

from __future__ import annotations

import base64
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Make the repo root importable so we can reuse the existing infer.py functions.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Reuse the project's existing detection API — do NOT reimplement.
from infer import (  # noqa: E402
    CLASS_NAMES,
    IMG_EXTS,
    PALETTE,
    draw_box,
    draw_count_panel,
    run_on_frame,
)
from ultralytics import YOLO  # noqa: E402

# ── configuration ────────────────────────────────────────────────────────────

# Prefer trained weights; fall back to the shipped pre-trained ones.
_DEFAULT_WEIGHTS = _REPO_ROOT / "runs" / "surgical" / "weights" / "best.pt"
_PRETRAINED     = _REPO_ROOT / "pre-trained_weights" / "best.pt"
WEIGHTS_PATH    = Path(os.environ.get("WEIGHTS", _DEFAULT_WEIGHTS))
if not WEIGHTS_PATH.exists():
    WEIGHTS_PATH = _PRETRAINED

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".flv"}
MAX_BYTES_IMAGE = 25 * 1024 * 1024   # 25 MB
MAX_BYTES_VIDEO = 250 * 1024 * 1024  # 250 MB

# Try these codecs in order — the first that opens a writer is used.
# H.264 variants play in every modern browser; mp4v is the fallback.
_VIDEO_CODECS = ["avc1", "H264", "mp4v"]


# ── model singleton ──────────────────────────────────────────────────────────

_model: YOLO | None = None


def get_model() -> YOLO:
    global _model
    if _model is None:
        if not WEIGHTS_PATH.exists():
            raise RuntimeError(f"Weights not found: {WEIGHTS_PATH}")
        print(f"[webapp] Loading model from {WEIGHTS_PATH} …")
        _model = YOLO(str(WEIGHTS_PATH))
        print("[webapp] Model ready.")
    return _model


# ── video helpers ────────────────────────────────────────────────────────────

def _open_writer(path: Path, w: int, h: int, fps: float):
    """Open a VideoWriter using the first working codec from _VIDEO_CODECS."""
    for fourcc_str in _VIDEO_CODECS:
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
        if writer.isOpened():
            return writer
        writer.release()
    # Last-resort: mp4v always opens even if playback is limited.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, fps, (w, h))


def _cleanup(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except OSError:
        pass


# ── app ──────────────────────────────────────────────────────────────────────

app = FastAPI(title="Surgical Instrument Detector", version="1.0.0")

STATIC_DIR = Path(__file__).resolve().parent / "static"
TEST_IMAGES_DIR = _REPO_ROOT / "test_images"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
# Serve the sample images so the frontend can render thumbnails.
if TEST_IMAGES_DIR.exists():
    app.mount("/test-images", StaticFiles(directory=TEST_IMAGES_DIR), name="test-images")


@app.get("/api/test-images")
def test_images():
    """List the sample images shipped under ``test_images/`` at the repo root."""
    if not TEST_IMAGES_DIR.exists():
        return {"images": []}
    images = [
        {
            "name": p.name,
            "url": f"/test-images/{p.name}",
        }
        for p in sorted(TEST_IMAGES_DIR.iterdir())
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    ]
    return {"images": images}


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "weights": str(WEIGHTS_PATH),
        "weights_found": WEIGHTS_PATH.exists(),
        "classes": CLASS_NAMES,
    }


@app.post("/api/infer/image")
async def infer_image(
    file: UploadFile = File(...),
    conf: float = Query(0.35, ge=0.0, le=1.0),
    iou: float = Query(0.45, ge=0.0, le=1.0),
):
    raw = await _read_upload(file, MAX_BYTES_IMAGE, IMG_EXTS, "image")

    # cv2.imdecode expects a numpy uint8 array.
    frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "Could not decode image. Unsupported or corrupt file.")

    model = get_model()
    frame, counts = run_on_frame(model, frame, conf, iou)
    draw_count_panel(frame, counts)

    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise HTTPException(500, "Failed to encode annotated image.")
    data_uri = "data:image/jpeg;base64," + base64.b64encode(buf).decode("ascii")

    return {
        "image": data_uri,
        "counts": dict(counts),
        "total": int(sum(counts.values())),
    }


@app.post("/api/infer/video")
async def infer_video(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    conf: float = Query(0.35, ge=0.0, le=1.0),
    iou: float = Query(0.45, ge=0.0, le=1.0),
):
    raw = await _read_upload(file, MAX_BYTES_VIDEO, VIDEO_EXTS, "video")

    tmp_in = Path(tempfile.mkstemp(suffix=_safe_suffix(file.filename))[1])
    tmp_out = Path(tempfile.mkstemp(suffix=".mp4")[1])
    tmp_in.write_bytes(raw)

    cap = cv2.VideoCapture(str(tmp_in))
    if not cap.isOpened():
        cap.release()
        _cleanup(tmp_in)
        _cleanup(tmp_out)
        raise HTTPException(400, "Could not open video. Unsupported or corrupt file.")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    writer = _open_writer(tmp_out, w, h, fps)

    model = get_model()
    counts: Counter = Counter()
    processed = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame, frame_counts = run_on_frame(model, frame, conf, iou)
            draw_count_panel(frame, frame_counts, fps)
            counts.update(frame_counts)
            writer.write(frame)
            processed += 1
    finally:
        cap.release()
        writer.release()
        _cleanup(tmp_in)

    if processed == 0:
        _cleanup(tmp_out)
        raise HTTPException(400, "No frames could be read from the video.")

    summary = {
        "counts": dict(counts),
        "total": int(sum(counts.values())),
        "frames": processed,
        "fps": round(fps, 2),
    }

    download_name = "annotated_" + (Path(file.filename).stem or "video") + ".mp4"
    background.add_task(_cleanup, tmp_out)

    return FileResponse(
        path=str(tmp_out),
        media_type="video/mp4",
        filename=download_name,
        headers={"X-Counts": json.dumps(summary)},
    )


async def _read_upload(file: UploadFile, max_bytes: int, allowed_exts: set, kind: str) -> bytes:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in allowed_exts:
        raise HTTPException(
            415,
            f"Unsupported file type '{ext}'. Allowed for {kind}: "
            f"{', '.join(sorted(allowed_exts))}",
        )
    raw = await file.read()
    if len(raw) == 0:
        raise HTTPException(400, "Empty file.")
    if len(raw) > max_bytes:
        raise HTTPException(413, f"File too large. Max {max_bytes // (1024*1024)} MB.")
    return raw


def _safe_suffix(filename: str | None) -> str:
    ext = Path(filename or "").suffix.lower()
    return ext if ext in VIDEO_EXTS else ".mp4"


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")
