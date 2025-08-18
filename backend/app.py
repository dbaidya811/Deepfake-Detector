from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Any

from .detector import analyze_image
from .schemas import DetectionResult, DetectionFeatures

app = FastAPI(title="AI vs Original Image Detector")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/detect", response_model=DetectionResult)
async def detect(file: UploadFile = File(...)) -> Any:
    content = await file.read()
    result_dict = analyze_image(content)

    features = DetectionFeatures(**result_dict["features"])  # type: ignore[arg-type]
    result = DetectionResult(
        is_ai_generated=result_dict["is_ai_generated"],
        confidence=result_dict["confidence"],
        reasons=result_dict["reasons"],
        features=features,
    )
    return JSONResponse(content=result.dict())
