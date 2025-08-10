import os
import tempfile
from typing import Dict
import traceback
import logging
import json
import numpy as np
import cv2

from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from backend.inference.video import analyze_video_heuristic, generate_xai_samples
from backend.inference.audio import analyze_audio_heuristic
try:
    from backend.inference.video_onnx import analyze_video_onnx
    ONNX_AVAILABLE = True
except Exception:
    ONNX_AVAILABLE = False
try:
    from backend.models.manager import check_and_update_models
    MODEL_MGR_AVAILABLE = True
except Exception:
    MODEL_MGR_AVAILABLE = False

app = FastAPI(title="Deepfake Detector (CPU-only MVP)")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("deepfake-detector")

# CORS (allow local file-based frontends and same-origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static frontend under /static and index at /
FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
async def serve_index():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Frontend not found"}


def make_verdict(score: float) -> str:
    if score >= 0.75:
        return "Likely Deepfake"
    if score >= 0.45:
        return "Suspicious"
    return "Likely Real"


def make_ai_generated(score: float) -> bool:
    # Mark as AI-generated for any non-real verdict (>= Suspicious threshold)
    return score >= 0.45


def score_confidence(score: float) -> str:
    if score >= 0.85:
        return "very_high"
    if score >= 0.70:
        return "high"
    if score >= 0.55:
        return "medium"
    if score >= 0.40:
        return "low"
    return "very_low"


# ==========================
# Startup: optional model update
# ==========================
@app.on_event("startup")
def on_startup_models():
    if MODEL_MGR_AVAILABLE:
        try:
            res = check_and_update_models()
            logger.info("Model update check: %s", res)
        except Exception:
            logger.exception("Model update check failed")


@app.post("/analyze/video")
async def analyze_video(file: UploadFile = File(...), xai: bool = Query(False)) -> Dict:
    if not file.filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".webm")):
        raise HTTPException(status_code=400, detail="Unsupported video format")
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        # Try ONNX model first if present
        if ONNX_AVAILABLE:
            try:
                onnx_result = analyze_video_onnx(tmp_path, model_path=os.path.join(os.path.dirname(__file__), "..", "models", "video.onnx"))
            except Exception:
                onnx_result = None

            if isinstance(onnx_result, dict) and "score" in onnx_result:
                onnx_result["engine"] = "onnx"
                onnx_result["verdict"] = make_verdict(onnx_result.get("score", 0.5))
                onnx_result["ai_generated"] = make_ai_generated(onnx_result.get("score", 0.5))
                onnx_result["confidence_label"] = score_confidence(onnx_result.get("score", 0.5))
                if xai:
                    # attach heuristic XAI overlays as a proxy (cheap and model-agnostic)
                    try:
                        onnx_result.setdefault("details", {})
                        onnx_result["details"]["xai"] = generate_xai_samples(tmp_path, max_samples=3)
                    except Exception:
                        pass
                return JSONResponse(onnx_result)

        # Fallback to heuristic
        result = analyze_video_heuristic(tmp_path, xai=xai)
        result["engine"] = "heuristic"
        result["verdict"] = make_verdict(result.get("score", 0.5))
        result["ai_generated"] = make_ai_generated(result.get("score", 0.5))
        result["confidence_label"] = score_confidence(result.get("score", 0.5))
        return JSONResponse(result)
    except Exception as e:
        logger.exception("Video analysis failed")
        return JSONResponse({"error": str(e), "trace": traceback.format_exc()[-500:]}, status_code=400)
    finally:
        try:
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


# ==========================
# Admin: trigger model update
# ==========================
@app.post("/admin/update_models")
async def admin_update_models() -> Dict:
    if not MODEL_MGR_AVAILABLE:
        return JSONResponse({"error": "model_manager_unavailable"}, status_code=400)
    try:
        res = check_and_update_models()
        return JSONResponse({"ok": True, "result": res})
    except Exception as e:
        logger.exception("Admin model update failed")
        return JSONResponse({"error": str(e)}, status_code=500)


# ==========================
# Report/appeal endpoint
# ==========================
@app.post("/report")
async def submit_report(payload: dict) -> Dict:
    try:
        os.makedirs(os.path.join(os.path.dirname(__file__), "logs"), exist_ok=True)
        log_path = os.path.join(os.path.dirname(__file__), "logs", "reports.jsonl")
        record = {
            "timestamp": __import__("datetime").datetime.utcnow().isoformat() + "Z",
            "payload": payload,
        }
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record) + "\n")
        return JSONResponse({"ok": True})
    except Exception as e:
        logger.exception("Report write failed")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/analyze/fusion")
async def analyze_fusion(
    video: UploadFile = File(...),
    audio: UploadFile = File(...),
    xai: bool = Query(False),
    v_weight: float = Query(0.7),
    a_weight: float = Query(0.3),
) -> Dict:
    if not video.filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".webm")):
        return JSONResponse({"error": "Unsupported video format"}, status_code=400)
    if not audio.filename.lower().endswith((".wav", ".mp3", ".flac", ".m4a", ".ogg")):
        return JSONResponse({"error": "Unsupported audio format"}, status_code=400)

    # normalize weights
    try:
        w_sum = max(v_weight + a_weight, 1e-6)
        v_w = float(v_weight) / w_sum
        a_w = float(a_weight) / w_sum
    except Exception:
        v_w, a_w = 0.7, 0.3

    tmp_v_path = None
    tmp_a_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video.filename)[1]) as tv:
            tv.write(await video.read())
            tmp_v_path = tv.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio.filename)[1]) as ta:
            ta.write(await audio.read())
            tmp_a_path = ta.name

        # Visual path: ONNX if available else heuristic
        vis = None
        if ONNX_AVAILABLE:
            try:
                vis = analyze_video_onnx(tmp_v_path, model_path=os.path.join(os.path.dirname(__file__), "..", "models", "video.onnx"))
                if isinstance(vis, dict):
                    vis["engine"] = "onnx"
            except Exception:
                vis = None
        if not isinstance(vis, dict):
            vis = analyze_video_heuristic(tmp_v_path, xai=xai)
            vis["engine"] = "heuristic"

        aud = analyze_audio_heuristic(tmp_a_path)

        v_score = float(vis.get("score", 0.5))
        a_score = float(aud.get("score", 0.5))
        fused = float(v_w * v_score + a_w * a_score)

        payload = {
            "score": fused,
            "verdict": make_verdict(fused),
            "ai_generated": make_ai_generated(fused),
            "confidence_label": score_confidence(fused),
            "engine": "fusion",
            "weights": {"visual": v_w, "audio": a_w},
            "visual": vis,
            "audio": aud,
        }
        return JSONResponse(payload)
    except Exception as e:
        logger.exception("Fusion analysis failed")
        return JSONResponse({"error": str(e), "trace": traceback.format_exc()[-500:]}, status_code=400)
    finally:
        for p in (tmp_v_path, tmp_a_path):
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass


@app.post("/analyze/audio")
async def analyze_audio(file: UploadFile = File(...)) -> Dict:
    if not file.filename.lower().endswith((".wav", ".mp3", ".flac", ".m4a", ".ogg")):
        raise HTTPException(status_code=400, detail="Unsupported audio format")
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        try:
            result = analyze_audio_heuristic(tmp_path)
            result["verdict"] = make_verdict(result.get("score", 0.5))
            return JSONResponse(result)
        except Exception as e:
            logger.exception("Audio analysis failed")
            return JSONResponse(
                status_code=400,
                content={
                    "error": "audio_analysis_failed",
                    "message": str(e),
                    "trace": traceback.format_exc().splitlines()[-3:],
                },
            )
    finally:
        try:
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


@app.get("/health")
async def health():
    return {"status": "ok"}
