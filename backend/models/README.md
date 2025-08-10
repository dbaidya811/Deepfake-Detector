# Models (Optional, for higher accuracy)

This app supports optional ONNX models for higher-accuracy deepfake detection on CPU. If a model is found, the backend will use it; otherwise it falls back to fast heuristics.

## Video model
- Expected path: `backend/models/video.onnx`
- The backend will try to use it via ONNX Runtime (CPUExecutionProvider).
- Input is assumed to be a single RGB image tensor (NCHW or NHWC). The code performs ImageNet-style normalization and takes the last value of the output vector (or the single output) as the "deepfake probability". You can adapt this if your model differs.

## How to enable ONNX inference
1) Install ONNX Runtime (CPU):
   - PowerShell (inside venv):
     ```powershell
     pip install onnxruntime
     ```
   - Note: If you're on a very new Python version and wheels are not yet available, this may fail. In that case, keep using the heuristic engine for now, or let me know and I can suggest alternatives.
2) Place your model file at `backend/models/video.onnx`.
3) Restart the server. The `/analyze/video` endpoint will report `engine: "onnx"` in its JSON when the model is used.

## Audio model (future)
- A similar `audio.onnx` pipeline can be added. If you have a specific spoofing model, share its input/output spec and I will wire it up.

## Tips
- For best results, use a face-focused classifier trained on deepfake datasets (e.g., FaceForensics++, DFDC). Preprocess frames by cropping faces before classification for higher accuracy.
- If your model expects face crops, we can add a face detector (e.g., RetinaFace/MTCNN) and run the model on face chips. This improves results but adds dependencies and compute.
