import json
import os
import random
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import cv2
import numpy as np
from fastapi import HTTPException, UploadFile, status

from core.image_utils import ImageUtils
from core.ensemble_inference import EnsembleWBF

HISTORY_PATH = Path("web/history.json")
ALLOWED_MODES = {"ssd", "yolo", "ensemble"}
ALLOWED_MIME = {"image/jpeg", "image/png"}
MAX_UPLOAD_MB = 8


class DetectAPI:
    """
    Facade over the detector + utility helpers so other services (simulation, routes) reuse the same logic.
    """

    def __init__(self, detector, history_path: Path | str = HISTORY_PATH, patches_dir: Path | str = "data/test_images/images"):
        self.detector = detector
        self.history_path = Path(history_path)
        self.patches_dir = Path(patches_dir)
        self.ensemble = EnsembleWBF()

    def validate_input(self, file: UploadFile, mode: str):
        self.validate_mode(mode)
        if file.content_type not in ALLOWED_MIME:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Only JPEG/PNG are supported",
            )

    def validate_mode(self, mode: str):
        if mode not in ALLOWED_MODES:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"mode must be one of {sorted(ALLOWED_MODES)}",
            )

    def decode_image(self, content: bytes):
        img = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Corrupted image data",
            )
        return img

    def save_bytes_as_jpg(self, content: bytes, folder: str) -> str:
        os.makedirs(folder, exist_ok=True)
        filename = f"{uuid.uuid4()}.jpg"
        path = os.path.join(folder, filename)
        img = self.decode_image(content)
        cv2.imwrite(path, img)
        return path

    def save_image(self, img, folder: str) -> str:
        os.makedirs(folder, exist_ok=True)
        filename = f"{uuid.uuid4()}.jpg"
        path = os.path.join(folder, filename)
        cv2.imwrite(path, img)
        return path

    def detect_api(self, image_bytes: bytes, model: str, draw_all: bool = False) -> Dict:
        self.validate_mode(model)
        img = self.decode_image(image_bytes)
        preds = self.detector.predict(img, mode=model)
        if model == "ensemble" and "ensemble" not in preds:
            preds["ensemble"] = self.ensemble.predict(
                preds.get("ssd", []),
                preds.get("yolo", []),
                img.shape[1],
                img.shape[0],
            )
        result_img = self.draw_predictions(img, preds, mode=model, draw_all=draw_all)
        return {"predictions": preds, "image": result_img, "model": model}

    def load_history(self):
        if not self.history_path.exists():
            return []
        try:
            with open(self.history_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def save_history(self, history):
        os.makedirs(self.history_path.parent, exist_ok=True)
        with open(self.history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=4)

    def add_history_entry(self, model: str, threshold: float, original_url: str, result_url: str):
        history = self.load_history()
        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": model,
            "threshold": float(threshold),
            "original": original_url,
            "result": result_url,
        }
        history.insert(0, entry)
        history = history[:20]
        self.save_history(history)

    def list_patches(self, count: int = 6):
        images_dir = Path(self.patches_dir)
        if not images_dir.exists():
            raise HTTPException(status_code=404, detail="Images directory not found")

        files = [p.name for p in images_dir.glob("*.jpg")]
        if not files:
            raise HTTPException(status_code=404, detail="No images found")

        random.shuffle(files)
        selected = files[: max(1, min(count, len(files)))]

        origin_lat, origin_lon = 48.5, 32.25
        patches = []
        for idx, name in enumerate(selected):
            row = idx // 3
            col = idx % 3
            lat = origin_lat + (row * 0.0006) + random.uniform(-0.0002, 0.0002)
            lon = origin_lon + (col * 0.0006) + random.uniform(-0.0002, 0.0002)
            patches.append(
                {
                    "label": f"Patch {idx+1}",
                    "url": f"/test_images/{name}",
                    "lat": lat,
                    "lon": lon,
                }
            )
        return {"origin": {"lat": origin_lat, "lon": origin_lon}, "patches": patches}

    def get_history(self):
        return self.load_history()

    def draw_predictions(self, img, preds, mode: str, draw_all: bool):
        if draw_all:
            return ImageUtils.draw_multiple_models(img, preds)

        boxes = preds.get(mode, [])
        return ImageUtils.draw_boxes(
            img,
            boxes,
            color=(0, 0, 255) if mode == "ensemble" else (0, 255, 0),
            label=mode.upper(),
        )
