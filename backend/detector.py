from __future__ import annotations

import io
import math
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ExifTags


def _variance_of_laplacian(img_gray: np.ndarray) -> float:
    # Normalize to [0,1]
    if img_gray.max() > 1.0:
        img_gray = img_gray / 255.0
    # Fast 3x3 Laplacian via numpy (4-neighbor):
    # kernel = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
    g = img_gray
    g_pad = np.pad(g, 1, mode="reflect")
    center = g_pad[1:-1, 1:-1]
    up = g_pad[0:-2, 1:-1]
    down = g_pad[2:, 1:-1]
    left = g_pad[1:-1, 0:-2]
    right = g_pad[1:-1, 2:]
    lap = (-4.0) * center + up + down + left + right
    return float(lap.var())


def _high_frequency_ratio(img_gray: np.ndarray) -> float:
    # FFT-based high-frequency energy ratio
    f = np.fft.fft2(img_gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    h, w = img_gray.shape
    cy, cx = h // 2, w // 2
    radius = min(h, w) // 8  # low-frequency radius

    yy, xx = np.ogrid[:h, :w]
    mask_low = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2

    total_energy = magnitude.sum() + 1e-8
    low_energy = magnitude[mask_low].sum()
    high_energy = total_energy - low_energy
    return float(high_energy / total_energy)


def _saturation_stats(img_rgb: np.ndarray) -> Tuple[float, float]:
    # Convert to HSV and compute S mean and entropy
    eps = 1e-12
    # RGB [0,255]
    rgb = img_rgb.astype(np.float32) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    cmax = np.max(rgb, axis=-1)
    cmin = np.min(rgb, axis=-1)
    delta = cmax - cmin
    saturation = np.where(cmax == 0, 0.0, delta / (cmax + eps))

    s_mean = float(np.mean(saturation))
    # entropy of histogram
    hist, _ = np.histogram(saturation, bins=32, range=(0.0, 1.0), density=True)
    p = hist / (hist.sum() + eps)
    entropy = float(-np.sum(np.where(p > 0, p * np.log2(p + eps), 0.0)) / math.log2(32))  # normalized 0..1
    return s_mean, entropy


def _resize_for_analysis(img: Image.Image, max_side: int = 768) -> Image.Image:
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        new_size = (max(64, int(w * scale)), max(64, int(h * scale)))
        return img.resize(new_size, Image.BICUBIC)
    return img


def _exif_details(im: Image.Image) -> Tuple[bool, str | None, str | None, str | None, bool | None]:
    make = model = lens = None
    gps_present: bool | None = None
    try:
        exif = im.getexif()
        if exif:
            exif_present = len(exif.items()) > 0
            inv = {v: k for k, v in ExifTags.TAGS.items()}
            make = exif.get(inv.get("Make"))
            model = exif.get(inv.get("Model"))
            lens = exif.get(inv.get("LensModel")) or exif.get(inv.get("LensMake"))
            gps_tag = inv.get("GPSInfo")
            gps_present = gps_tag in exif if gps_tag is not None else False
            return bool(exif_present), str(make) if make else None, str(model) if model else None, str(lens) if lens else None, bool(gps_present)
    except Exception:
        pass
    return False, None, None, None, None


def _noise_std_highpass(img_gray: np.ndarray) -> float:
    # High-pass residual using 5x5 Gaussian blur subtraction
    g = img_gray.astype(np.float32)
    if g.max() > 1.0:
        g = g / 255.0
    # separable gaussian kernel approx
    k = np.array([1, 4, 6, 4, 1], dtype=np.float32)
    k = k / k.sum()
    # horizontal blur
    gpad = np.pad(g, ((0, 0), (2, 2)), mode="reflect")
    hb = (
        gpad[:, 0:-4] * k[0]
        + gpad[:, 1:-3] * k[1]
        + gpad[:, 2:-2] * k[2]
        + gpad[:, 3:-1] * k[3]
        + gpad[:, 4:] * k[4]
    )
    # vertical blur
    gpad2 = np.pad(hb, ((2, 2), (0, 0)), mode="reflect")
    vb = (
        gpad2[0:-4, :] * k[0]
        + gpad2[1:-3, :] * k[1]
        + gpad2[2:-2, :] * k[2]
        + gpad2[3:-1, :] * k[3]
        + gpad2[4:, :] * k[4]
    )
    hp = g - vb
    return float(np.std(hp))


def _blockiness_score(img_gray: np.ndarray) -> float:
    # Measure 8x8 JPEG-like block boundary artifacts using boundary gradient spikes
    g = img_gray.astype(np.float32)
    if g.max() > 1.0:
        g = g / 255.0
    h, w = g.shape
    # vertical boundaries at multiples of 8
    cols = [c for c in range(8, w, 8)]
    rows = [r for r in range(8, h, 8)]
    if not cols and not rows:
        return 0.0
    # compute abs gradient across boundaries
    score_parts: List[float] = []
    for c in cols:
        left = g[:, c - 1]
        right = g[:, c]
        score_parts.append(float(np.mean(np.abs(right - left))))
    for r in rows:
        up = g[r - 1, :]
        down = g[r, :]
        score_parts.append(float(np.mean(np.abs(down - up))))
    # normalize by global gradient level
    gy, gx = np.gradient(g)
    baseline = float(np.mean(np.abs(gx)) + np.mean(np.abs(gy)) + 1e-6)
    return float(np.mean(score_parts) / baseline)


def _fft_checkerboard_score(img_gray: np.ndarray) -> float:
    # Strength of alternating pixel pattern (checkerboard) using modulation with (-1)^(x+y)
    g = img_gray.astype(np.float32)
    if g.max() > 1.0:
        g = g / 255.0
    h, w = g.shape
    yy, xx = np.mgrid[0:h, 0:w]
    alt = ((-1.0) ** (xx + yy)).astype(np.float32)
    mod = g * alt
    # energy of modulated image relative to original
    e_orig = float(np.mean(np.square(g)) + 1e-8)
    e_mod = float(np.mean(np.square(mod)))
    return float(e_mod / e_orig)


def _sobel_gradients(g: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Simple Sobel filters
    if g.max() > 1.0:
        g = g / 255.0
    kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
    ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
    gp = np.pad(g, 1, mode="reflect")
    def conv(k: np.ndarray) -> np.ndarray:
        return (
            gp[0:-2, 0:-2] * k[0, 0]
            + gp[0:-2, 1:-1] * k[0, 1]
            + gp[0:-2, 2:] * k[0, 2]
            + gp[1:-1, 0:-2] * k[1, 0]
            + gp[1:-1, 1:-1] * k[1, 1]
            + gp[1:-1, 2:] * k[1, 2]
            + gp[2:, 0:-2] * k[2, 0]
            + gp[2:, 1:-1] * k[2, 1]
            + gp[2:, 2:] * k[2, 2]
        )
    gx = conv(kx)
    gy = conv(ky)
    return gy, gx


def _lighting_direction_entropy(img_gray: np.ndarray) -> float:
    gy, gx = _sobel_gradients(img_gray.astype(np.float32))
    ang = np.arctan2(gy, gx)  # -pi..pi
    # histogram of angles
    hist, _ = np.histogram(ang, bins=36, range=(-math.pi, math.pi), density=True)
    p = hist / (hist.sum() + 1e-12)
    ent = float(-np.sum(np.where(p > 0, p * np.log2(p + 1e-12), 0.0)) / math.log2(36))
    return ent


def _optional_face_metrics(img_rgb: np.ndarray) -> Tuple[int, float | None]:
    # Try OpenCV Haar Cascade if available; otherwise return no faces
    try:
        import cv2  # type: ignore

        gray = (0.2989 * img_rgb[..., 0] + 0.5870 * img_rgb[..., 1] + 0.1140 * img_rgb[..., 2]).astype(np.uint8)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(32, 32))
        faces = list(faces) if faces is not None else []
        sym_scores: List[float] = []
        for (x, y, w, h) in faces[:3]:  # up to 3 faces
            crop = gray[y : y + h, x : x + w]
            crop = cv2.resize(crop, (128, 128), interpolation=cv2.INTER_AREA)
            # left-right symmetry via SSIM-like measure
            left = crop[:, :64].astype(np.float32) / 255.0
            right = np.fliplr(crop[:, 64:]).astype(np.float32) / 255.0
            mu_l, mu_r = left.mean(), right.mean()
            var_l, var_r = left.var() + 1e-6, right.var() + 1e-6
            cov = ((left - mu_l) * (right - mu_r)).mean()
            ssim_like = (2 * mu_l * mu_r + 0.01) * (2 * cov + 0.03) / ((mu_l**2 + mu_r**2 + 0.01) * (var_l + var_r + 0.03))
            sym_scores.append(float(ssim_like))
        faces_detected = len(faces)
        face_symmetry = float(np.mean(sym_scores)) if sym_scores else None
        return faces_detected, face_symmetry
    except Exception:
        return 0, None


def analyze_image(image_bytes: bytes) -> Dict:
    # Load image
    with Image.open(io.BytesIO(image_bytes)) as im:
        im = im.convert("RGB")
        exif_present, exif_make, exif_model, exif_lens, exif_gps = _exif_details(im)

        im_small = _resize_for_analysis(im)
        rgb = np.array(im_small)
        # grayscale
        gray = np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.float32)

        vol = _variance_of_laplacian(gray)
        hfr = _high_frequency_ratio(gray)
        s_mean, s_entropy = _saturation_stats(rgb)
        noise_std_hp = _noise_std_highpass(gray)
        blockiness = _blockiness_score(gray)
        fft_check = _fft_checkerboard_score(gray)
        light_entropy = _lighting_direction_entropy(gray)
        faces_detected, face_sym = _optional_face_metrics(rgb)

        # Heuristic scoring (0=Original, 1=AI). Tuned empirically.
        score = 0.0
        reasons: List[str] = []

        # EXIF absence slightly increases AI score
        if not exif_present:
            score += 0.15
            reasons.append("No/limited EXIF metadata detected (common in generated images)")
        else:
            reasons.append(
                "EXIF metadata present (common in camera photos)"
                + (f" — {exif_make or ''} {exif_model or ''}".strip())
            )
            if exif_gps:
                reasons.append("GPS data found in EXIF (often present in real photos)")

        # Very low high-frequency ratio and low Laplacian variance -> smoother -> more AI
        if hfr < 0.80:
            delta = (0.80 - hfr) * 0.4  # up to +0.32
            score += delta
            reasons.append(f"Lower high-frequency energy ({hfr:.2f}) suggests synthetic smoothness")
        else:
            reasons.append(f"Rich high-frequency detail ({hfr:.2f}) suggests natural capture")

        if vol < 0.0015:
            delta = min(0.25, (0.0015 - vol) * 200)  # scale small numbers
            score += delta
            reasons.append(f"Very low edge variance ({vol:.4f}) indicates possible AI denoising")
        else:
            reasons.append(f"Edge variance ({vol:.4f}) consistent with natural texture")

        # Saturation stats: extremely uniform or extreme saturation may indicate synthesis
        if s_entropy < 0.75:
            score += 0.15
            reasons.append(f"Low saturation entropy ({s_entropy:.2f}) suggests uniform synthetic colors")
        if s_mean > 0.6:
            score += 0.1
            reasons.append(f"High mean saturation ({s_mean:.2f}) often seen in generated images")

        # Noise and compression: very uniform noise (low std) OR very high blockiness can indicate synthetic/compressed
        if noise_std_hp < 0.02:
            score += 0.1
            reasons.append(f"Very low high-pass noise std ({noise_std_hp:.3f}) suggests uniform synthetic noise")
        else:
            reasons.append(f"High-pass noise std ({noise_std_hp:.3f}) consistent with sensor noise")

        if blockiness > 1.8:
            score += 0.1
            reasons.append(f"Strong 8x8 blockiness ({blockiness:.2f}) indicates compression/synthesis artifacts")

        # FFT checkerboard artifacts
        if fft_check > 1.1:
            score += 0.12
            reasons.append(f"Checkerboard-like frequency pattern ({fft_check:.2f}) suggests upsampling artifacts")

        # Faces and symmetry
        if faces_detected > 0 and face_sym is not None:
            if face_sym < 0.6:
                score += 0.12
                reasons.append(f"Face symmetry score low ({face_sym:.2f}) — possible AI/deepfake anomalies")
            else:
                reasons.append(f"Face symmetry score ({face_sym:.2f}) looks normal")

        # Lighting/shadow consistency: extremely high entropy of gradient directions may indicate chaotic lighting
        if light_entropy > 0.95:
            score += 0.08
            reasons.append(f"Very diffuse gradient orientation (entropy {light_entropy:.2f}) suggests inconsistent lighting")

        # Clamp and convert to decision
        score = float(max(0.0, min(1.0, score)))
        is_ai = score >= 0.5

        return {
            "is_ai_generated": is_ai,
            "confidence": score if is_ai else (1.0 - score),
            "reasons": reasons,
            "features": {
                "exif_present": exif_present,
                "exif_camera_make": exif_make,
                "exif_camera_model": exif_model,
                "exif_lens": exif_lens,
                "exif_gps_present": exif_gps,
                "variance_of_laplacian": float(vol),
                "high_freq_ratio": float(hfr),
                "saturation_mean": float(s_mean),
                "saturation_entropy": float(s_entropy),
                "noise_std_hp": float(noise_std_hp),
                "blockiness_score": float(blockiness),
                "fft_checkerboard_score": float(fft_check),
                "faces_detected": int(faces_detected),
                "face_symmetry_score": float(face_sym) if face_sym is not None else None,
                "lighting_direction_entropy": float(light_entropy),
                "image_width": im.width,
                "image_height": im.height,
                "mode": "RGB",
            },
        }
