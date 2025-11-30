import cv2
import numpy as np
import time


# ============================================================
#  Measure time decorator (for speed tests)
# ============================================================

def measure_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        return result, end - start
    return wrapper


# ============================================================
#  Draw boxes of a single model
# ============================================================

def draw_boxes(img, boxes, color=(0, 255, 0), label="model", thickness=2):
    out = img.copy()

    for (x1, y1, x2, y2, score) in boxes:
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)),
                      color, thickness)

        cv2.putText(
            out,
            f"{label}: {score:.2f}",
            (int(x1), int(y1) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    return out


# ============================================================
#  Draw multiple models at once
# ============================================================

COLORS = {
    "ssd": (255, 0, 0),       # blue
    "yolo": (0, 255, 0),      # green
    "ensemble": (0, 0, 255),  # red
}


def draw_multiple_models(img, predictions, thickness=2):
    out = img.copy()

    for key, boxes in predictions.items():
        if key not in COLORS:
            continue

        color = COLORS[key]

        for (x1, y1, x2, y2, score) in boxes:
            cv2.rectangle(out, (int(x1), int(y1)),
                          (int(x2), int(y2)), color, thickness)
            cv2.putText(
                out,
                f"{key}: {score:.2f}",
                (int(x1), int(y1) - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    return out


# ============================================================
#  Resize while keeping aspect ratio
# ============================================================

def resize_keep_ratio(img, max_size=800):
    h, w = img.shape[:2]
    scale = max_size / max(h, w)

    if scale >= 1:
        return img

    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h))


# ============================================================
#  Save 2x2 grid: Original | SSD / YOLO | Ensemble
# ============================================================

def save_comparison_grid(img, preds, save_path="comparison.jpg"):
    ssd_img = draw_boxes(img, preds.get("ssd", []),
                         COLORS["ssd"], "SSD")
    yolo_img = draw_boxes(img, preds.get("yolo", []),
                          COLORS["yolo"], "YOLO")
    ens_img = draw_boxes(img, preds.get("ensemble", []),
                         COLORS["ensemble"], "ENS")

    # Create 2x2 grid
    top = np.hstack([img, ssd_img])
    bottom = np.hstack([yolo_img, ens_img])
    grid = np.vstack([top, bottom])

    cv2.imwrite(save_path, grid)
    return grid

def draw_preds_and_gt(img, pred_boxes, gt_boxes,
                      pred_color=(0, 255, 0), gt_color=(0, 255, 255),
                      pred_label="pred", gt_label="gt"):
    """
    pred_boxes: [[x1, y1, x2, y2, score], ...]
    gt_boxes:   [[x1, y1, x2, y2], ...]
    """
    out = img.copy()

    # GT boxes
    for (x1, y1, x2, y2) in gt_boxes:
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)),
                      gt_color, 2)
        cv2.putText(out, gt_label,
                    (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, gt_color, 2)

    # Predicted boxes
    for (x1, y1, x2, y2, score) in pred_boxes:
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)),
                      pred_color, 2)
        cv2.putText(out, f"{pred_label}: {score:.2f}",
                    (int(x1), int(y1) - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, pred_color, 2)

    return out

# ============================================================
#                     TEST SECTION
# ============================================================

if __name__ == "__main__":
    from detector import Detector
    print("Running utils.py tests...")

    # Load any test image (change path if needed)
    img_path = "E:\\Code\\Mag_diploma\\Landmine\\data\\test_images\\images\\5_Zone_1_Mine_1cm_depth__2-2m_TemperatureInCenter__33-C_jpg.rf.94dcca373102acf10657071ac661716d.jpg"
    img = cv2.imread(img_path)

    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    
    det = Detector()

    pred = det.predict(img, mode="ensemble")

    # Fake sample boxes for testing
    ssd_boxes = pred['ssd']
    yolo_boxes = pred['yolo']
    ens_boxes = pred['ensemble']

    # --- Test 1: draw_boxes ---
    out1 = draw_boxes(img, ssd_boxes, color=(255, 0, 0), label="SSD")
    cv2.imwrite("test_draw_ssd.jpg", out1)
    print("✓ draw_boxes test saved -> test_draw_ssd.jpg")

    # --- Test 2: draw_multiple_models ---
    preds = {
        "ssd": ssd_boxes,
        "yolo": yolo_boxes,
        "ensemble": ens_boxes
    }
    out2 = draw_multiple_models(img, preds)
    cv2.imwrite("test_draw_multiple.jpg", out2)
    print("✓ draw_multiple_models test saved -> test_draw_multiple.jpg")

    # --- Test 3: save_comparison_grid ---
    grid = save_comparison_grid(img, preds, save_path="test_grid.jpg")
    print("✓ comparison grid saved -> test_grid.jpg")

    # --- Test 4: resize_keep_ratio ---
    resized = resize_keep_ratio(img, max_size=400)
    cv2.imwrite("test_resize.jpg", resized)
    print("✓ resize_keep_ratio test saved -> test_resize.jpg")

    print("\nAll utils tests completed successfully.")
