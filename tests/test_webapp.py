"""End-to-end tests for the FastAPI web app (model stubbed, fully offline)."""
from __future__ import annotations

import cv2
import numpy as np
from fastapi.testclient import TestClient

import webapp.main as app_module


def _jpeg_bytes() -> bytes:
    img = np.zeros((64, 64, 3), np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def _client(monkeypatch, model=None):
    # Avoid loading the real YOLO weights during tests.
    monkeypatch.setattr(app_module, "get_model", lambda: model)
    return TestClient(app_module.app)


def test_health(monkeypatch):
    client = _client(monkeypatch)
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert len(body["classes"]) == 26
    # pre-trained_weights/best.pt ships with the repo.
    assert body["weights_found"] is True


def test_index_serves_frontend(monkeypatch):
    client = _client(monkeypatch)
    r = client.get("/")
    assert r.status_code == 200
    assert "<html" in r.text.lower()
    assert "Surgical Instrument Detector" in r.text


def test_static_assets(monkeypatch):
    client = _client(monkeypatch)
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/styles.css").status_code == 200


def test_video_panel_is_not_hidden_by_default(monkeypatch):
    """Regression: clicking the Video tab showed nothing because the panel
    carried a `hidden` class whose `display:none !important` beat
    `.panel.active`. The panel must rely solely on `.panel` / `.panel.active`.
    """
    client = _client(monkeypatch)
    html = client.get("/").text
    # Locate the video panel tag and assert it has no `hidden` class.
    idx = html.index('id="panel-video"')
    tag = html[idx - 40 : idx + 80]
    assert 'class="panel"' in tag or 'class="panel active"' in tag
    assert "hidden" not in tag


def test_app_js_toggles_hidden_class(monkeypatch):
    """The tab switcher must remove `hidden` from the activated panel so
    visibility is robust regardless of the initial markup."""
    client = _client(monkeypatch)
    js = client.get("/static/app.js").text
    assert 'classList.toggle("hidden"' in js


def test_infer_image(monkeypatch, fake_model):
    client = _client(monkeypatch, model=fake_model)
    files = {"file": ("img.jpg", _jpeg_bytes(), "image/jpeg")}
    r = client.post("/api/infer/image", files=files)
    assert r.status_code == 200
    body = r.json()
    assert body["image"].startswith("data:image/jpeg;base64,")
    assert body["counts"] == {"Needle Holder": 1}
    assert body["total"] == 1


def test_infer_image_no_detections(monkeypatch, empty_model):
    client = _client(monkeypatch, model=empty_model)
    files = {"file": ("img.png", _jpeg_bytes(), "image/jpeg")}
    r = client.post("/api/infer/image", files=files)
    assert r.status_code == 200
    assert r.json()["total"] == 0


def test_infer_image_rejects_bad_extension(monkeypatch):
    client = _client(monkeypatch)
    files = {"file": ("file.txt", b"hello", "text/plain")}
    r = client.post("/api/infer/image", files=files)
    assert r.status_code == 415


def test_infer_image_rejects_empty_file(monkeypatch):
    client = _client(monkeypatch)
    files = {"file": ("empty.jpg", b"", "image/jpeg")}
    r = client.post("/api/infer/image", files=files)
    assert r.status_code == 400


def test_infer_image_conf_validation(monkeypatch):
    client = _client(monkeypatch)
    files = {"file": ("img.jpg", _jpeg_bytes(), "image/jpeg")}
    # conf > 1.0 violates the Query(ge=0.0, le=1.0) constraint → 422.
    r = client.post("/api/infer/image", files=files, params={"conf": 1.5})
    assert r.status_code == 422
