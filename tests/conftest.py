"""Shared test fixtures.

The real YOLO weights are heavy and need torch + a model file, so the inference
helpers are exercised against a tiny in-memory stub that quacks like
``ultralytics.YOLO`` (just enough of ``model.predict(...)[0].boxes`` for
``infer.run_on_frame``).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Make the repo root importable so `import infer` / `import webapp.main` work.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


class FakeBox:
    def __init__(self, cls_id: int, conf: float, xyxy: tuple[int, int, int, int]):
        self.cls = np.array([cls_id], dtype=np.float32)
        self.conf = np.array([conf], dtype=np.float32)
        self.xyxy = np.array([xyxy], dtype=np.float32)


class FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class FakeModel:
    """Minimal stand-in for ``ultralytics.YOLO`` used by ``run_on_frame``."""

    def __init__(self, boxes):
        self._boxes = boxes

    def predict(self, frame, conf=None, iou=None, verbose=False):
        return [FakeResult(self._boxes)]


@pytest.fixture
def fake_model():
    # One "Needle Holder" (class 0) detection, confidence 0.90.
    return FakeModel([FakeBox(cls_id=0, conf=0.90, xyxy=(10, 10, 50, 50))])


@pytest.fixture
def empty_model():
    """A model that detects nothing."""
    return FakeModel([])
