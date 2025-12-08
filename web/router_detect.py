import json
import os
import uuid
from datetime import datetime
from pathlib import Path
import random

import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, Form, HTTPException, status
from fastapi.responses import JSONResponse

from core.utils import draw_boxes, draw_multiple_models
from web.models import detector, set_thresholds

HISTORY_PATH = "web/history.json"
MAX_HISTORY = 20
ALLOWED_MODES = {"ssd", "yolo", "ensemble"}
ALLOWED_MIME = {"image/jpeg", "image/png"}
MAX_UPLOAD_MB = 8

router = APIRouter()


def load_history():
    if not os.path.exists(HISTORY_PATH):
        return []
    try:
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_history(history):
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=4)


def add_history_entry(model: str, threshold: float, original_url: str, result_url: str):
    history = load_history()
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model,
        "threshold": float(threshold),
        "original": original_url,
        "result": result_url,
    }
    history.insert(0, entry)
    if len(history) > MAX_HISTORY:
        history = history[:MAX_HISTORY]
    save_history(history)


def validate_input(file: UploadFile, mode: str):
    if mode not in ALLOWED_MODES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"mode must be one of {sorted(ALLOWED_MODES)}",
        )
    if file.content_type not in ALLOWED_MIME:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only JPEG/PNG are supported",
        )


def decode_image(content: bytes):
    img = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Corrupted image data",
        )
    return img


def save_bytes_as_jpg(content: bytes, folder: str) -> str:
    os.makedirs(folder, exist_ok=True)
    filename = f"{uuid.uuid4()}.jpg"
    path = os.path.join(folder, filename)
    img = decode_image(content)
    cv2.imwrite(path, img)
    return path


@router.post("/detect")
async def detect_api(
    file: UploadFile,
    mode: str = Form("ensemble"),
    threshold: float = Form(0.4),
    draw_all: bool = Form(False),
):
    validate_input(file, mode)
    content = await file.read()
    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds {MAX_UPLOAD_MB} MB limit",
        )

    set_thresholds(threshold, threshold)
    img = decode_image(content)

    original_path = save_bytes_as_jpg(content, "web/static/uploaded")
    original_url = f"/static/uploaded/{os.path.basename(original_path)}"

    preds = detector.predict(img, mode=mode)

    if draw_all:
        result_img = draw_multiple_models(img, preds)
    else:
        boxes = preds.get(mode, [])
        result_img = draw_boxes(
            img,
            boxes,
            label=mode.upper(),
            color=(0, 0, 255) if mode == "ensemble" else (0, 255, 0),
        )

    os.makedirs("web/static/results", exist_ok=True)
    result_filename = f"{uuid.uuid4()}.jpg"
    result_path = os.path.join("web/static/results", result_filename)
    cv2.imwrite(result_path, result_img)
    result_url = f"/static/results/{result_filename}"

    add_history_entry(
        model=mode,
        threshold=threshold,
        original_url=original_url,
        result_url=result_url,
    )

    return JSONResponse(
        {
            "status": "ok",
            "image_url": result_url,
            "original_url": original_url,
            "predictions": preds,
        },
        status_code=status.HTTP_200_OK,
    )


@router.get("/history")
def get_history():
    history = load_history()
    return history


@router.get("/patches")
def list_patches(count: int = 6):
    """
    Return a list of sample patches from data/test_images/images with pseudo-GPS coords.
    """
    images_dir = Path("data/test_images/images")
    if not images_dir.exists():
        raise HTTPException(status_code=404, detail="Images directory not found")

    files = [p.name for p in images_dir.glob("*.jpg")]
    if not files:
        raise HTTPException(status_code=404, detail="No images found")

    random.shuffle(files)
    selected = files[:max(1, min(count, len(files)))]

    origin_lat, origin_lon = 48.5, 32.25
    patches = []
    for idx, name in enumerate(selected):
        row = idx // 3
        col = idx % 3
        lat = origin_lat + (row * 0.0006) + random.uniform(-0.0002, 0.0002)
        lon = origin_lon + (col * 0.0006) + random.uniform(-0.0002, 0.0002)
        patches.append({
            "label": f"Патч {idx+1}",
            "url": f"/test_images/{name}",
            "lat": lat,
            "lon": lon,
        })

    return {"origin": {"lat": origin_lat, "lon": origin_lon}, "patches": patches}
