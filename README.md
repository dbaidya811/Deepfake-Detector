# AI vs Original Image Detector

A minimal full‑stack app to heuristically detect whether an uploaded image is AI‑generated or a real camera photo.

- Frontend: HTML/CSS/JS (static), modern glass UI, two‑column layout (photo left, result right) with animations.
- Backend: FastAPI (Python) with image analysis features.

Frontend: HTML/CSS/JS (static), modern glass UI, two-column layout (photo left, result right) with animations.

## Features (Heuristics)
- EXIF details: camera make/model, lens, GPS presence.
- Texture/detail: variance of Laplacian, high‑frequency energy ratio (FFT).
- Color: saturation mean + entropy.
- Noise/compression: high‑pass noise std, JPEG‑like blockiness.
- Frequency artifacts: checkerboard score.
- Faces (optional): Haar Cascade face detection + symmetry score (needs OpenCV).
- Lighting consistency: gradient‑direction entropy.

Returned fields are documented in `backend/schemas.py` (`DetectionFeatures`, `DetectionResult`).

## Project Structure
```
backend/
  app.py           # FastAPI app (routes: /api/health, /api/detect)
  detector.py      # Image analysis + scoring (no heavy deps; OpenCV optional)
  schemas.py       # Pydantic response models
  requirements.txt # Backend dependencies
frontend/
  index.html       # UI
  styles.css       # Modern glass theme
  app.js           # Upload & show results
  favicon.svg      # App icon
```

## Prerequisites
- Python 3.13 recommended (project was set up/tested with 3.13).
- Windows PowerShell commands below assume project root: `c:\Users\dbaid\OneDrive\Desktop\New folder`.

## Setup & Run (Windows PowerShell)
1) Create & activate virtualenv
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
2) Upgrade pip and install deps
```
python -m pip install --upgrade pip
pip install -r backend\requirements.txt
```
3) (Optional) Enable face detection/symmetry
```
pip install opencv-python
```
4) Start the API server
```
python -m uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000
```
5) Open the UI
- Double‑click `frontend/index.html` or open it in your browser.
- Upload an image and click Analyze.

The frontend calls the API at `http://127.0.0.1:8000`. If you change the port/host, edit `frontend/app.js` → `API_BASE`.

## API
- Health
```
GET /api/health -> {"status":"ok"}
```
- Detect
```
POST /api/detect
Content-Type: multipart/form-data (file=<image>)
Response: DetectionResult (see backend/schemas.py)
```
Example (PowerShell, using Invoke-WebRequest):
```
Invoke-WebRequest -Uri http://127.0.0.1:8000/api/detect -Method POST -InFile .\sample.jpg -ContentType "multipart/form-data" -OutFile result.json
```

## Notes & Limitations
- Heuristics are not definitive. Highly compressed, edited, or filtered real photos can look synthetic—and vice versa.
- EXIF may be stripped by platforms; absence doesn’t guarantee AI.
- OpenCV is optional; without it, face features will be skipped.
- If running behind a different host/port, update `API_BASE` in `frontend/app.js`.

## Troubleshooting
- Module not found / version errors
  - Ensure you installed from `backend/requirements.txt`.
  - For Python 3.13, this repo pins `numpy>=2.1` and `pillow>=10.4`.
- CORS/Network issues
  - CORS is open in `backend/app.py` for local usage.
- OpenCV install is heavy
  - Skip it if you don’t need face symmetry; the app still works without it.

## License
For educational/demo use. No warranty. Use responsibly.
