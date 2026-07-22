"""Unit tests for the drawing / inference helpers in ``infer.py``."""
from __future__ import annotations

from collections import Counter

import cv2
import numpy as np

import infer


def test_class_and_palette_lengths():
    assert len(infer.CLASS_NAMES) == 26
    assert len(infer.PALETTE) == len(infer.CLASS_NAMES)


def test_img_exts_contains_common_formats():
    assert {".jpg", ".jpeg", ".png", ".bmp", ".webp"} <= infer.IMG_EXTS


def test_has_gui_is_a_bool():
    # Probe must not raise even on headless builds.
    assert isinstance(infer.HAS_GUI, bool)


def test_draw_box_annotates_frame():
    frame = np.zeros((100, 100, 3), np.uint8)
    infer.draw_box(frame, 10, 10, 50, 50, "Scalpel", 0.9, (0, 255, 0))
    assert frame.any()  # rectangle / label were drawn


def test_draw_count_panel_mutates_frame():
    frame = np.zeros((200, 200, 3), np.uint8)
    counts = Counter({"Scalpel": 1, "Needle Holder": 2})
    infer.draw_count_panel(frame, counts, fps=30.0)
    assert frame.any()


def test_run_on_frame_returns_counts_and_shape(fake_model):
    frame = np.zeros((120, 120, 3), np.uint8)
    out, counts = infer.run_on_frame(fake_model, frame, 0.35, 0.45)
    assert counts["Needle Holder"] == 1
    assert out.shape == frame.shape
    assert out.any()  # annotated


def test_run_on_frame_no_detections(empty_model):
    # No boxes → empty counts, frame left untouched by drawing.
    frame = np.zeros((64, 64, 3), np.uint8)
    out, counts = infer.run_on_frame(empty_model, frame, 0.35, 0.45)
    assert counts == Counter()
    assert not out.any()
