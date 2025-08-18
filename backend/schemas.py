from pydantic import BaseModel
from typing import List, Dict, Any


class DetectionFeatures(BaseModel):
    exif_present: bool
    exif_camera_make: str | None = None
    exif_camera_model: str | None = None
    exif_lens: str | None = None
    exif_gps_present: bool | None = None
    variance_of_laplacian: float
    high_freq_ratio: float
    saturation_mean: float
    saturation_entropy: float
    noise_std_hp: float
    blockiness_score: float
    fft_checkerboard_score: float
    faces_detected: int
    face_symmetry_score: float | None = None
    lighting_direction_entropy: float
    image_width: int
    image_height: int
    mode: str


class DetectionResult(BaseModel):
    is_ai_generated: bool
    confidence: float  # 0..1
    reasons: List[str]
    features: DetectionFeatures
