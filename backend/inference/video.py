import cv2
import numpy as np
from typing import Dict
import base64


def _frame_highfreq_energy(gray: np.ndarray) -> float:
    # High-pass filter via Laplacian to estimate high-frequency detail
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(np.mean(np.abs(lap)))


def _frame_blockiness(gray: np.ndarray) -> float:
    # Simple blockiness heuristic (JPEG-like compression artifacts)
    # compute gradient along 8x8 boundaries
    h, w = gray.shape
    # Column boundaries (… 7|8 … 15|16 …)
    v_left = gray[:, 7::8].astype(np.float32)
    v_right = gray[:, 8::8].astype(np.float32)
    v_m = min(v_left.shape[1] if v_left.ndim == 2 else 0,
              v_right.shape[1] if v_right.ndim == 2 else 0)
    if v_m > 0:
        v_edges = v_left[:, :v_m] - v_right[:, :v_m]
        v_mean = np.mean(np.abs(v_edges))
    else:
        v_mean = 0.0

    # Row boundaries (… 7|8 … 15|16 …)
    h_top = gray[7::8, :].astype(np.float32)
    h_bot = gray[8::8, :].astype(np.float32)
    h_m = min(h_top.shape[0] if h_top.ndim == 2 else 0,
              h_bot.shape[0] if h_bot.ndim == 2 else 0)
    if h_m > 0:
        h_edges = h_top[:h_m, :] - h_bot[:h_m, :]
        h_mean = np.mean(np.abs(h_edges))
    else:
        h_mean = 0.0

    return float(v_mean + h_mean)


def _temporal_inconsistency(prev: np.ndarray, curr: np.ndarray) -> float:
    # Mean absolute frame difference (normalized)
    diff = cv2.absdiff(prev, curr)
    return float(np.mean(diff) / 255.0)


def _detect_face_roi(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Detect largest face using Haar cascades; return ROI (BGR). Fallback to center crop if not found.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    # Use OpenCV built-in Haar cascades (no external download)
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    except Exception:
        face_cascade = None

    h, w = gray.shape
    if face_cascade is not None and not face_cascade.empty():
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        if len(faces) > 0:
            # pick largest
            x, y, fw, fh = max(faces, key=lambda r: r[2] * r[3])
            # pad a bit
            pad = int(0.1 * max(fw, fh))
            x0 = max(0, x - pad)
            y0 = max(0, y - pad)
            x1 = min(w, x + fw + pad)
            y1 = min(h, y + fh + pad)
            roi = frame_bgr[y0:y1, x0:x1]
            if roi.size > 0:
                return roi
    # fallback: center crop
    sz = min(h, w)
    cy, cx = h // 2, w // 2
    half = sz // 3  # take central third
    y0, y1 = max(0, cy - half), min(h, cy + half)
    x0, x1 = max(0, cx - half), min(w, cx + half)
    roi = frame_bgr[y0:y1, x0:x1]
    return roi if roi.size > 0 else frame_bgr


def _ela_score(bgr: np.ndarray, q: int = 90) -> float:
    """Error Level Analysis proxy: recompress and measure diff."""
    # Encode to JPEG in-memory
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(np.clip(q, 50, 95))]
    ok, enc = cv2.imencode('.jpg', bgr, encode_param)
    if not ok:
        return 0.0
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    diff = cv2.absdiff(bgr, dec)
    return float(np.mean(diff) / 255.0)


def _color_anomaly_score(bgr: np.ndarray) -> float:
    """Basic color feature: saturation variance and YCrCb chroma variance."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32) / 255.0
    s_var = float(np.var(sat))
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    cr = ycc[:, :, 1].astype(np.float32)
    cb = ycc[:, :, 2].astype(np.float32)
    c_var = float(np.var(cr) + np.var(cb)) / (255.0 ** 2)
    # Normalize roughly to [0,1]
    return float(np.clip(0.5 * (s_var / 0.05) + 0.5 * (c_var / 0.05), 0.0, 1.0))


def _to_base64_png(img_bgr: np.ndarray) -> str:
    ok, enc = cv2.imencode('.png', img_bgr)
    if not ok:
        return ""
    return base64.b64encode(enc.tobytes()).decode('ascii')


def _make_heatmap_overlay(roi_bgr: np.ndarray, gray: np.ndarray) -> np.ndarray:
    # Laplacian magnitude
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    lap_abs = np.abs(lap)
    lap_norm = cv2.normalize(lap_abs, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    lap_color = cv2.applyColorMap(lap_norm, cv2.COLORMAP_JET)
    # ELA map
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    ok, enc = cv2.imencode('.jpg', roi_bgr, encode_param)
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR) if ok else roi_bgr
    ela = cv2.absdiff(roi_bgr, dec)
    ela_gray = cv2.cvtColor(ela, cv2.COLOR_BGR2GRAY)
    ela_norm = cv2.normalize(ela_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    ela_color = cv2.applyColorMap(ela_norm, cv2.COLORMAP_TURBO)
    # Combine heatmaps
    combined = cv2.addWeighted(lap_color, 0.6, ela_color, 0.4, 0)
    overlay = cv2.addWeighted(roi_bgr, 0.5, combined, 0.5, 0)
    return overlay


def generate_xai_samples(video_path: str, max_samples: int = 3) -> Dict:
    samples = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"samples": samples}
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(int(round(fps / 1.0)), 1)
    idx = 0
    taken = 0
    while taken < max_samples:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step != 0:
            idx += 1
            continue
        idx += 1
        frame = cv2.resize(frame, (640, 360))
        roi = _detect_face_roi(frame)
        roi = cv2.resize(roi, (384, 384))
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        overlay = _make_heatmap_overlay(roi, gray)
        b64 = _to_base64_png(overlay)
        samples.append({"frame_index": idx, "overlay_png_b64": b64})
        taken += 1
    cap.release()
    return {"samples": samples}


def analyze_video_heuristic(video_path: str, xai: bool = False) -> Dict:
    """
    CPU-only lightweight heuristic analysis.
    Returns a score in [0,1] estimating likelihood of manipulation.

    Heuristics used:
    - Face-ROI based: high-frequency detail, blockiness
    - Irregular temporal inconsistency
    - ELA (recompression difference)
    - Color anomaly (saturation/chroma variance)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"score": 0.5, "details": {"error": "Cannot open video"}}

    sample_rate_fps = 1.0  # sample roughly 1 frame per second
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(int(round(fps / sample_rate_fps)), 1)

    idx = 0
    hf_vals, blk_vals, temp_vals, ela_vals, col_vals = [], [], [], [], []
    prev_gray = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step != 0:
            idx += 1
            continue
        idx += 1
        # Resize for speed and consistency
        frame = cv2.resize(frame, (640, 360))
        roi = _detect_face_roi(frame)
        roi = cv2.resize(roi, (384, 384))
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        hf_vals.append(_frame_highfreq_energy(gray))
        blk_vals.append(_frame_blockiness(gray))
        if prev_gray is not None:
            temp_vals.append(_temporal_inconsistency(prev_gray, gray))
        prev_gray = gray
        ela_vals.append(_ela_score(roi))
        col_vals.append(_color_anomaly_score(roi))

    cap.release()

    if len(hf_vals) < 3:
        # Not enough frames; return neutral-ish score
        details = {"frames": len(hf_vals)}
        if xai:
            details["xai"] = generate_xai_samples(video_path, max_samples=2)
        return {"score": 0.5, "details": details}

    # Normalize features robustly
    hf = np.array(hf_vals)
    blk = np.array(blk_vals)
    tmp = np.array(temp_vals) if len(temp_vals) else np.array([0.0])
    ela = np.array(ela_vals)
    col = np.array(col_vals)

    # Expected ranges (empirical defaults for heuristic)
    # Lower HF may indicate over-smoothing from generative models
    hf_norm = 1.0 - np.clip((hf - 2.0) / 6.0, 0.0, 1.0)  # low HF -> closer to 1
    # Higher blockiness may indicate re-encoding artifacts
    blk_norm = np.clip((blk - 1.0) / 6.0, 0.0, 1.0)
    # Irregular temporal changes may suggest blending/artifacts
    tmp_norm = np.clip((np.abs(tmp - np.median(tmp))) / (np.median(tmp) + 1e-6), 0.0, 2.0)
    tmp_norm = np.clip(tmp_norm / 2.0, 0.0, 1.0)
    # ELA higher -> more suspicious
    ela_norm = np.clip((ela - 0.02) / 0.1, 0.0, 1.0)
    # Color anomaly higher -> more suspicious
    col_norm = np.clip((col - 0.05) / 0.2, 0.0, 1.0)

    # Aggregate with weights
    score = float(
        0.35 * np.median(hf_norm) +
        0.20 * np.median(blk_norm) +
        0.20 * np.median(tmp_norm) +
        0.15 * np.median(ela_norm) +
        0.10 * np.median(col_norm)
    )
    details = {
        "frames": int(len(hf_vals)),
        "hf_median": float(np.median(hf)),
        "blk_median": float(np.median(blk)),
        "temp_median": float(np.median(tmp)) if len(temp_vals) else 0.0,
        "ela_median": float(np.median(ela)),
        "color_anom_median": float(np.median(col)),
    }
    if xai:
        details["xai"] = generate_xai_samples(video_path, max_samples=3)
    return {"score": float(np.clip(score, 0.0, 1.0)), "details": details}


def analyze_frame_heuristic(frame_bgr: np.ndarray) -> Dict:
    """
    Per-frame CPU-only heuristic for realtime.
    Uses face ROI + HF, blockiness, ELA, color anomaly.
    Returns {score, details}.
    """
    try:
        # Resize for speed/consistency
        frame = cv2.resize(frame_bgr, (640, 360))
        roi = _detect_face_roi(frame)
        roi = cv2.resize(roi, (384, 384))
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        hf = _frame_highfreq_energy(gray)
        blk = _frame_blockiness(gray)
        ela = _ela_score(roi)
        col = _color_anomaly_score(roi)

        # Normalize like video path (single-frame, no temporal)
        hf_norm = 1.0 - np.clip((hf - 2.0) / 6.0, 0.0, 1.0)
        blk_norm = np.clip((blk - 1.0) / 6.0, 0.0, 1.0)
        ela_norm = np.clip((ela - 0.02) / 0.1, 0.0, 1.0)
        col_norm = np.clip((col - 0.05) / 0.2, 0.0, 1.0)

        score = float(
            0.40 * hf_norm +
            0.25 * blk_norm +
            0.20 * ela_norm +
            0.15 * col_norm
        )
        details = {
            "hf": float(hf),
            "blk": float(blk),
            "ela": float(ela),
            "color_anom": float(col),
        }
        return {"score": float(np.clip(score, 0.0, 1.0)), "details": details}
    except Exception:
        return {"score": 0.5, "details": {"error": "frame_analysis_failed"}}
