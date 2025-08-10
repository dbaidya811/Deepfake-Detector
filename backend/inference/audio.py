import numpy as np
import librosa
from typing import Dict


def _safe_load(path: str, sr: int = 16000):
    y, sr = librosa.load(path, sr=sr, mono=True)
    # Trim silence
    yt, _ = librosa.effects.trim(y, top_db=40)
    if yt.size < sr:  # ensure at least 1s
        yt = y
    return yt.astype(np.float32), sr


def analyze_audio_heuristic(audio_path: str) -> Dict:
    """
    CPU-only lightweight heuristic analysis for audio spoofing.
    Returns a score in [0,1] estimating likelihood of synthetic audio.

    Heuristics (rule-of-thumb, not SOTA):
    - Spectral flatness and centroid variance
    - Over-smooth MFCC statistics
    - Abnormal zero-crossing rate and RMS dynamics
    """
    try:
        y, sr = _safe_load(audio_path, sr=16000)
    except Exception as e:
        return {"score": 0.5, "details": {"error": f"cannot load audio: {e}"}}

    # Frame parameters
    hop = 256
    win = 1024

    # Features
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=win, hop_length=hop)[0]
    rms = librosa.feature.rms(y=y, frame_length=win, hop_length=hop)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=win, hop_length=hop)[0]
    flatness = librosa.feature.spectral_flatness(y=y, n_fft=win, hop_length=hop)[0]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=win, hop_length=hop)

    # Normalize and compute statistics
    def robust_stats(x):
        return float(np.median(x)), float(np.mean(np.abs(x - np.median(x))))

    zcr_med, zcr_mad = robust_stats(zcr)
    rms_med, rms_mad = robust_stats(rms)
    cen_med, cen_mad = robust_stats(centroid)
    flat_med, flat_mad = robust_stats(flatness)

    # MFCC smoothness: low frame-to-frame diff may indicate over-smoothing
    mfcc_diff = np.mean(np.abs(np.diff(mfcc, axis=1)))

    # Map heuristics to [0,1] suspicion scores
    # Higher flatness -> noise-like -> sometimes indicates vocoder artifacts
    s_flat = np.clip((flat_med - 0.2) / 0.5, 0.0, 1.0)
    # Lower MFCC dynamics -> over-smoothing
    s_mfcc = np.clip((0.08 - mfcc_diff) / 0.08, 0.0, 1.0)
    # Low RMS dynamics may suggest uniform loudness
    s_rms = np.clip((0.02 - rms_mad) / 0.02, 0.0, 1.0)
    # Abnormal centroid variance
    s_cen = np.clip(cen_mad / (cen_med + 1e-6) * 2.0, 0.0, 1.0)
    # Abnormal zcr variance
    s_zcr = np.clip(zcr_mad / (zcr_med + 1e-6) * 3.0, 0.0, 1.0)

    score = float(0.35 * s_mfcc + 0.25 * s_flat + 0.2 * s_rms + 0.1 * s_cen + 0.1 * s_zcr)
    details = {
        "frames": int(len(rms)),
        "flat_med": flat_med,
        "mfcc_dyn": float(mfcc_diff),
        "rms_mad": rms_mad,
        "cen_mad": cen_mad,
        "zcr_mad": zcr_mad,
    }
    return {"score": float(np.clip(score, 0.0, 1.0)), "details": details}
