import os
from typing import Optional, Dict

import numpy as np
import cv2

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover
    ort = None


class VideoONNX:
    def __init__(self, model_path: str):
        if ort is None:
            raise RuntimeError("onnxruntime is not installed")
        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)
        so = ort.SessionOptions()
        so.intra_op_num_threads = max(1, os.cpu_count() or 1)
        so.inter_op_num_threads = 1
        providers = ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(model_path, sess_options=so, providers=providers)
        self.input = self.sess.get_inputs()[0]
        self.output = self.sess.get_outputs()[0]
        self.in_name = self.input.name
        self.out_name = self.output.name
        # Expect shape like (N, C, H, W) or (N, H, W, C)
        self.input_shape = tuple(dim if isinstance(dim, int) else 224 for dim in self.input.shape)
        self.channels_last = False
        if len(self.input_shape) == 4:
            n, d1, d2, d3 = self.input_shape
            # Heuristic: if second dim is small (3), treat as NCHW
            if d1 in (1, 3):
                self.channels_last = False
                self.height = int(d2)
                self.width = int(d3)
                self.channels = int(d1)
            else:
                self.channels_last = True
                self.height = int(d1)
                self.width = int(d2)
                self.channels = int(d3)
        else:
            # Fallback
            self.height = 224
            self.width = 224
            self.channels = 3

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        img = cv2.resize(frame, (self.width, self.height))
        img = img.astype(np.float32) / 255.0
        # Standard ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        if not self.channels_last:
            img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        img = np.expand_dims(img, 0)
        return img

    def predict_frame(self, frame_bgr: np.ndarray) -> float:
        # returns probability of deepfake (assumes model outputs that)
        x = self._preprocess(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        out = self.sess.run([self.out_name], {self.in_name: x})[0]
        # Heuristic to convert output to probability
        prob = float(out.reshape(-1)[-1]) if out.size > 1 else float(out.reshape(-1)[0])
        # Clip to [0,1]
        return float(np.clip(prob, 0.0, 1.0))


def analyze_video_onnx(video_path: str, model_path: str = "models/video.onnx") -> Optional[Dict]:
    if ort is None or not os.path.exists(model_path):
        return None
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Cannot open video"}

    clf = VideoONNX(model_path)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(int(round(fps / 1.0)), 1)  # 1 fps

    idx = 0
    probs = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step != 0:
            idx += 1
            continue
        idx += 1
        frame = cv2.resize(frame, (512, 288))
        probs.append(clf.predict_frame(frame))
    cap.release()

    if not probs:
        return {"score": 0.5, "details": {"frames": 0}}

    score = float(np.median(probs))
    return {"score": float(np.clip(score, 0.0, 1.0)), "details": {"frames": len(probs), "median_prob": score}}
