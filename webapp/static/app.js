// Surgical Instrument Detector — frontend logic (vanilla JS, no build step).
"use strict";

const API = {
  health: "/api/health",
  image:  "/api/infer/image",
  video:  "/api/infer/video",
};

// ── small helpers ──────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);

function toast(msg, isErr = false) {
  const el = $("toast");
  el.textContent = msg;
  el.classList.toggle("err", isErr);
  el.classList.add("show");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => el.classList.remove("show"), 4200);
}

function setLoading(btn, on, label) {
  if (on) {
    btn._orig = btn.innerHTML;
    btn.innerHTML = `<span class="spinner"></span> ${label || "Processing…"}`;
    btn.disabled = true;
  } else {
    btn.innerHTML = btn._orig || btn.innerHTML;
    btn.disabled = false;
  }
}

function chipHTML(name, n, color) {
  return `<span class="chip"><span class="dot" style="background:${color}"></span>${name}<span class="n">${n}</span></span>`;
}

// Derive a colour per class from the same palette idea as the backend.
function classColor(name) {
  const idx = (window._CLASS_NAMES || []).indexOf(name);
  const palette = window._PALETTE || [];
  if (idx >= 0 && palette[idx]) {
    const [b, g, r] = palette[idx]; // backend stores BGR
    return `rgb(${r},${g},${b})`;
  }
  // stable fallback hash
  let h = 0;
  for (let i = 0; i < name.length; i++) h = (h * 31 + name.charCodeAt(i)) >>> 0;
  return `hsl(${h % 360} 65% 55%)`;
}

function renderCounts(containerId, counts, total, extra) {
  const el = $(containerId);
  const entries = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  if (!entries.length) {
    el.innerHTML = `<h3>Detections</h3><p class="empty">No instruments detected. Try lowering the confidence threshold.</p>`;
    return;
  }
  const chips = entries
    .map(([name, n]) => chipHTML(name, n, classColor(name)))
    .join("");
  const extraHtml = extra ? `<span class="meta">${extra}</span>` : "";
  el.innerHTML = `<h3>Detections</h3><div class="chips"><span class="chip total">Total ${total}</span>${chips}</div>${extraHtml}`;
}

// ── tab switching ───────────────────────────────────────────────────
function setupTabs() {
  const tabs = [
    ["tab-image", "panel-image"],
    ["tab-video", "panel-video"],
  ];
  tabs.forEach(([tabId, panelId]) => {
    $(tabId).addEventListener("click", () => {
      tabs.forEach(([t, p]) => {
        $(t).classList.toggle("active", t === tabId);
        $(t).setAttribute("aria-selected", t === tabId ? "true" : "false");
        $(p).classList.toggle("active", p === panelId);
      });
    });
  });
}

// ── dropzone wiring (shared) ────────────────────────────────────────
function wireDropzone(dropId, inputId, onFile) {
  const drop = $(dropId);
  const input = $(inputId);
  drop.addEventListener("click", () => input.click());
  drop.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); }
  });
  input.addEventListener("change", () => {
    if (input.files[0]) onFile(input.files[0]);
  });
  ["dragenter", "dragover"].forEach((ev) =>
    drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("drag"); })
  );
  ["dragleave", "drop"].forEach((ev) =>
    drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("drag"); })
  );
  drop.addEventListener("drop", (e) => {
    const f = e.dataTransfer.files[0];
    if (f) onFile(f);
  });
}

// ── range slider live labels ────────────────────────────────────────
function wireRange(rangeId, valueId) {
  const r = $(rangeId), v = $(valueId);
  r.addEventListener("input", () => { v.textContent = parseFloat(r.value).toFixed(2); });
}

// ── health check + fetch class names/palette from backend ──────────
async function checkHealth() {
  const badge = $("status");
  try {
    const res = await fetch(API.health);
    if (!res.ok) throw new Error("bad status");
    const data = await res.json();
    if (!data.weights_found) {
      badge.className = "status err";
      badge.textContent = "Weights missing — train the model first";
      toast("Model weights not found. Run train.py, then restart.", true);
      return;
    }
    window._CLASS_NAMES = data.classes || [];
    // Synthesise a palette matching infer.py (seeded). We only need colours for
    // display; the actual boxes are drawn server-side.
    window._PALETTE = makePalette(window._CLASS_NAMES.length);
    badge.className = "status ok";
    badge.textContent = "Backend ready";
  } catch (e) {
    badge.className = "status err";
    badge.textContent = "Backend offline";
    toast("Cannot reach backend. Is uvicorn running?", true);
  }
}

function makePalette(n) {
  // Mirror of infer.py: rng.integers(80, 220, size=(n,3)) seeded with 42.
  // Simple LCG isn't identical to numpy's BitGenerator, so just produce stable
  // distinct colours with a deterministic hash (display-only).
  const out = [];
  for (let i = 0; i < n; i++) {
    const hue = (i * 360 / n) % 360;
    const [r, g, b] = hslToRgb(hue / 360, 0.55, 0.55);
    out.push([b, g, r]); // store as BGR to match classColor()
  }
  return out;
}

function hslToRgb(h, s, l) {
  let r, g, b;
  if (s === 0) { r = g = b = l; }
  else {
    const hue2rgb = (p, q, t) => {
      if (t < 0) t += 1; if (t > 1) t -= 1;
      if (t < 1 / 6) return p + (q - p) * 6 * t;
      if (t < 1 / 2) return q;
      if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
      return p;
    };
    const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
    const p = 2 * l - q;
    r = hue2rgb(p, q, h + 1 / 3);
    g = hue2rgb(p, q, h);
    b = hue2rgb(p, q, h - 1 / 3);
  }
  return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}

// ── IMAGE flow ─────────────────────────────────────────────────────
let imageFile = null;

function onImageFile(file) {
  imageFile = file;
  $("img-original").src = URL.createObjectURL(file);
  $("img-results").classList.remove("hidden");
  $("img-run").disabled = false;
}

async function runImage() {
  if (!imageFile) return;
  const btn = $("img-run");
  setLoading(btn, true, "Detecting…");
  try {
    const conf = $("img-conf").value;
    const iou = $("img-iou").value;
    const fd = new FormData();
    fd.append("file", imageFile);
    const res = await fetch(`${API.image}?conf=${conf}&iou=${iou}`, { method: "POST", body: fd });
    if (!res.ok) {
      const t = await res.text();
      throw new Error(t || `HTTP ${res.status}`);
    }
    const data = await res.json();
    $("img-annotated").src = data.image;
    renderCounts("img-counts", data.counts, data.total);
    toast(`Done — ${data.total} instrument(s) detected.`);
  } catch (e) {
    toast("Image detection failed: " + e.message, true);
  } finally {
    setLoading(btn, false);
  }
}

// ── VIDEO flow ─────────────────────────────────────────────────────
let videoFile = null;

function onVideoFile(file) {
  videoFile = file;
  $("vid-run").disabled = false;
  $("vid-results").classList.add("hidden");
}

async function runVideo() {
  if (!videoFile) return;
  const btn = $("vid-run");
  setLoading(btn, true, "Processing video…");
  try {
    const conf = $("vid-conf").value;
    const iou = $("vid-iou").value;
    const fd = new FormData();
    fd.append("file", videoFile);
    const res = await fetch(`${API.video}?conf=${conf}&iou=${iou}`, { method: "POST", body: fd });
    if (!res.ok) {
      const t = await res.text();
      throw new Error(t || `HTTP ${res.status}`);
    }
    const blob = await res.blob();

    // Counts come back in a custom response header (JSON).
    let summary = { counts: {}, total: 0, frames: 0, fps: 0 };
    const hdr = res.headers.get("X-Counts");
    if (hdr) {
      try { summary = JSON.parse(hdr); } catch (_) { /* keep defaults */ }
    }

    const url = URL.createObjectURL(blob);
    $("vid-annotated").src = url;
    $("vid-download").href = url;
    $("vid-download").download = "annotated_" + videoFile.name.replace(/\.[^.]+$/, "") + ".mp4";
    $("vid-results").classList.remove("hidden");

    const extra = `${summary.frames} frames · ${summary.fps} fps`;
    renderCounts("vid-counts", summary.counts, summary.total, extra);
    toast(`Done — ${summary.total} instrument(s) across ${summary.frames} frames.`);
  } catch (e) {
    toast("Video processing failed: " + e.message, true);
  } finally {
    setLoading(btn, false);
  }
}

// ── init ───────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  setupTabs();
  wireDropzone("img-drop", "img-input", onImageFile);
  wireDropzone("vid-drop", "vid-input", onVideoFile);
  wireRange("img-conf", "img-conf-v");
  wireRange("img-iou",  "img-iou-v");
  wireRange("vid-conf", "vid-conf-v");
  wireRange("vid-iou",  "vid-iou-v");
  $("img-run").addEventListener("click", runImage);
  $("vid-run").addEventListener("click", runVideo);
  checkHealth();
});
