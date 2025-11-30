import uuid
import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, Form
from core.utils import draw_boxes, draw_multiple_models
from web.models import detector, set_thresholds
import json
import os
from datetime import datetime

HISTORY_PATH = "web/history.json"
MAX_HISTORY = 20 

router = APIRouter()

def load_history():
    """Зчитати історію з файлу."""
    if not os.path.exists(HISTORY_PATH):
        return []
    try:
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_history(history):
    """Зберегти історію в файл."""
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=4)


def add_history_entry(model: str, threshold: float,
                      original_url: str, result_url: str):
    """Додати новий запис до історії."""
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


@router.post("/detect")
async def detect_api(
    file: UploadFile,
    mode: str = Form("ensemble"),
    threshold: float = Form(0.4),
    draw_all: bool = Form(False)
):
    
    # -------------------------
    # 1. Оновлюємо поріг
    # -------------------------
    set_thresholds(threshold, threshold)

    # -------------------------
    # 2. Читаємо фото
    # -------------------------
    img_bytes = await file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    original_name = f"{uuid.uuid4()}_original.jpg"
    original_path = f"web/static/uploaded/{original_name}"
    os.makedirs("web/static/uploaded", exist_ok=True)
    cv2.imwrite(original_path, img)

    # -------------------------
    # 3. Детекція
    # -------------------------
    preds = detector.predict(img, mode=mode)

    # -------------------------
    # 4. Обробка логіки відображення
    # -------------------------
    if draw_all:
        # Показати всі моделі
        result_img = draw_multiple_models(img, preds)
    else:
        # Показати тільки вибрану модель
        boxes = preds.get(mode, [])
        result_img = draw_boxes(
            img,
            boxes,
            label=mode.upper(),
            color=(0, 0, 255) if mode == "ensemble" else (0, 255, 0)
        )

    # -------------------------
    # 5. Зберігаємо зображення
    # -------------------------
    filename = f"{uuid.uuid4()}.jpg"
    save_path = f"web/static/results/{filename}"
    os.makedirs("web/static/results", exist_ok=True)
    cv2.imwrite(save_path, result_img)

    result_url = f"/static/results/{filename}"
    original_url = f"/static/uploaded/{original_name}"

    # 🔹 ДОДАЄМО В ІСТОРІЮ
    add_history_entry(
        model=mode,
        threshold=threshold,
        original_url=original_url,
        result_url=result_url
    )

    return {
        "status": "ok",
        "image_url": result_url,
        "original_url": original_url,
        "predictions": preds
    }

@router.get("/history")
def get_history():
    """Повернути список останніх детекцій."""
    history = load_history()
    return history
