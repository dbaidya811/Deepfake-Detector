# Deepfake Detector (CPU-only MVP)

Web app (HTML/CSS) + FastAPI backend for heuristic deepfake detection on video and audio. CPU-only friendly.

## Run (Windows)

1. Create venv and install deps:
   ```powershell
   python -m venv venv
   ./venv/Scripts/Activate.ps1
   pip install -r backend/requirements.txt
   ```
2. Start backend:
   ```powershell
   ./run_backend.bat
   ```
3. Open the app:
   - Visit http://127.0.0.1:8000/ in your browser

## Notes
- This MVP uses heuristics, not SOTA models; treat results as indicative only.
- Supported video: mp4, mov, avi, mkv, webm
- Supported audio: wav, mp3, flac, m4a, ogg

## Run with Docker (optional)

1. Build image:
   ```bash
   docker build -t deepfake-detector .
   ```
2. Run container:
   ```bash
   docker run --rm -p 8000:8000 deepfake-detector
   ```
3. Open http://127.0.0.1:8000/

## API Endpoints

- GET `/` — serves `frontend/index.html`
- GET `/health` — service health check
- POST `/analyze/video` — form-data: `file` (video); query: `xai` (bool)
- POST `/analyze/audio` — form-data: `file` (audio)
- POST `/analyze/fusion` — form-data: `video`, `audio`; query: `xai` (bool), `v_weight` (float), `a_weight` (float)
- POST `/report` — JSON body with report/appeal payload

## Frontend

- Static assets are served from `frontend/` under `/static`.
- UI tweaks: cards spaced vertically, Report/Appeal buttons right-aligned, and styled file inputs.

## Project Structure

```
project/
├─ app.py                  # Entrypoint exposing FastAPI app
├─ backend/
│  ├─ app.py               # Routes and logic
│  ├─ inference/           # Heuristic/ONNX analyzers
│  └─ models/              # Optional model manager/files
├─ frontend/
│  ├─ index.html           # UI (video & audio cards)
│  └─ style.css            # Theming and component styles
├─ run_backend.bat         # Windows runner
├─ requirements.txt        # Docker/root requirements
└─ Dockerfile              # CPU image
```

## Troubleshooting

- If video/audio parsing fails, ensure ffmpeg/libsndfile are available (installed in Docker image; on Windows, rely on Python packages).
- Large files may take time on CPU; try smaller inputs when testing.
- If `uvicorn` port is busy, change `--port` or stop the other service.
