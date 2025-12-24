import os
import uuid
from fastapi import APIRouter, UploadFile, Form, HTTPException, status
from fastapi.responses import JSONResponse

from core.detect_api import DetectAPI, MAX_UPLOAD_MB
from web.models import detector, set_thresholds

router = APIRouter()
detect_api_service = DetectAPI(detector)


@router.post("/detect")
async def detect_api(
    file: UploadFile,
    mode: str = Form("ensemble"),
    threshold: float = Form(0.4),
    draw_all: bool = Form(False),
):
    detect_api_service.validate_input(file, mode)
    content = await file.read()
    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds {MAX_UPLOAD_MB} MB limit",
        )

    set_thresholds(threshold, threshold)
    original_path = detect_api_service.save_bytes_as_jpg(content, "web/static/uploaded")
    original_url = f"/static/uploaded/{os.path.basename(original_path)}"

    result = detect_api_service.detect_api(content, model=mode, draw_all=draw_all)
    preds = result["predictions"]
    result_img = result["image"]

    os.makedirs("web/static/results", exist_ok=True)
    result_filename = f"{uuid.uuid4()}.jpg"
    result_path = os.path.join("web/static/results", result_filename)
    import cv2

    cv2.imwrite(result_path, result_img)
    result_url = f"/static/results/{result_filename}"

    detect_api_service.add_history_entry(
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
    return detect_api_service.get_history()


@router.get("/patches")
def list_patches(count: int = 6):
    """
    Return a list of sample patches from data/test_images/images with pseudo-GPS coords.
    """
    return detect_api_service.list_patches(count=count)
