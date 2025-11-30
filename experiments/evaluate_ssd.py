import os
import sys

# Add root folder to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import json
import cv2
import numpy as np

from core.detector import Detector
from experiments.metrics import iou, precision_recall_f1, average_precision_from_scores
from core.utils import draw_preds_and_gt


# ================================
# Load YOLO ground truth annotation
# ================================

def load_yolo_label(label_path, orig_w, orig_h):
    """
    YOLO TXT format:
    class x_center y_center width height   (normalized)
    Return: [x1, y1, x2, y2]
    """
    if not os.path.exists(label_path):
        return []

    boxes = []
    with open(label_path, "r") as f:
        for line in f.readlines():
            cls, x, y, w, h = map(float, line.split())

            # Convert from normalized YOLO to absolute box coordinates
            x1 = (x - w/2) * orig_w
            y1 = (y - h/2) * orig_h
            x2 = (x + w/2) * orig_w
            y2 = (y + h/2) * orig_h

            boxes.append([x1, y1, x2, y2])

    return boxes


# ================================
# Main evaluation
# ================================

def evaluate_ssd():
    detector = Detector()   # loads SSD + YOLO, but we use only SSD mode

    images_dir = "data/test_images/images"
    labels_dir = "data/test_images/labels"

    results_dir = "data/results/ssd/"
    vis_dir = os.path.join(results_dir, "vis")

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    iou_list = []
    tp = fp = fn = 0

    ap_scores = []
    all_scores = []
    all_labels = []

    for filename in os.listdir(images_dir):
        if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img_path = os.path.join(images_dir, filename)
        label_path = os.path.join(labels_dir, filename.replace(".jpg", ".txt"))

        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]

        # Load ground truth
        gt_boxes = load_yolo_label(label_path, w, h)

        # SSD prediction
        preds = detector.predict(img, mode="ssd")["ssd"]

        # Save visualization
        vis = draw_preds_and_gt(img, preds, gt_boxes, (255, 0, 0), pred_label="SSD")
        cv2.imwrite(os.path.join(vis_dir, filename), vis)

        # Evaluate IoU for each GT
        matched = set()
        for pred in preds:
            p_box = pred[:4]
            p_score = pred[4]

            best_iou = 0
            best_gt = -1

            for idx, gt_box in enumerate(gt_boxes):
                curr_iou = iou(p_box, gt_box)
                if curr_iou > best_iou:
                    best_iou = curr_iou
                    best_gt = idx

            if best_iou > 0.5:
                tp += 1
                matched.add(best_gt)
                iou_list.append(best_iou)

                all_scores.append(p_score)
                all_labels.append(1)
            else:
                fp += 1
                all_scores.append(p_score)
                all_labels.append(0)

        # Count false negatives
        for idx in range(len(gt_boxes)):
            if idx not in matched:
                fn += 1

    # Compute final metrics
    precision, recall, f1 = precision_recall_f1(tp, fp, fn)
    total_gt = tp + fn
    ap = average_precision_from_scores(all_scores, all_labels, total_gt)
    mean_iou = np.mean(iou_list) if iou_list else 0
    map50 = ap   # AP@0.5 threshold

    # Save metrics
    metrics = {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "IoU_mean": mean_iou,
        "mAP50": map50,
    }

    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    print("SSD evaluation completed.")
    print(json.dumps(metrics, indent=4))


if __name__ == "__main__":
    evaluate_ssd()
