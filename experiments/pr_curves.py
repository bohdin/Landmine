import os
import sys
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Add project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from core.detector import Detector
from experiments.metrics import iou, precision_recall_f1


def load_yolo_label(label_path, orig_w, orig_h):
    """Load ground truth YOLO annotations."""
    if not os.path.exists(label_path):
        return []

    boxes = []
    with open(label_path, "r") as f:
        for line in f.readlines():
            cls, x, y, w, h = map(float, line.split())

            x1 = (x - w/2) * orig_w
            y1 = (y - h/2) * orig_h
            x2 = (x + w/2) * orig_w
            y2 = (y + h/2) * orig_h

            boxes.append([x1, y1, x2, y2])

    return boxes


def evaluate_model(preds_all, gt_boxes, thresholds):
    """
    preds_all = {"ssd": [...], "yolo": [...], "ensemble": [...]}
    gt_boxes = [[x1,y1,x2,y2], ...]
    """
    stats = {model: {thr: {"tp": 0, "fp": 0, "fn": 0} for thr in thresholds}
             for model in ["ssd", "yolo", "ensemble"]}

    for model_name in ["ssd", "yolo", "ensemble"]:
        preds_model = preds_all[model_name]

        for thr in thresholds:
            preds = [p for p in preds_model if p[4] >= thr]
            matched = set()

            for pred in preds:
                p_box = pred[:4]
                best_iou = 0
                best_gt = -1

                for idx, gt_box in enumerate(gt_boxes):
                    curr_iou = iou(p_box, gt_box)
                    if curr_iou > best_iou:
                        best_iou = curr_iou
                        best_gt = idx

                if best_iou > 0.5:
                    stats[model_name][thr]["tp"] += 1
                    matched.add(best_gt)
                else:
                    stats[model_name][thr]["fp"] += 1

            for idx in range(len(gt_boxes)):
                if idx not in matched:
                    stats[model_name][thr]["fn"] += 1

    return stats


def compute_pr_curves(stats, thresholds):
    """
    Convert TP/FP/FN → Precision/Recall curves for each model.
    """
    curves = {}

    for model in ["ssd", "yolo", "ensemble"]:
        rows = []

        for thr in thresholds:
            s = stats[model][thr]
            tp, fp, fn = s["tp"], s["fp"], s["fn"]

            precision, recall, f1 = precision_recall_f1(tp, fp, fn)

            rows.append({
                "threshold": thr,
                "precision": precision,
                "recall": recall,
                "f1": f1
            })

        curves[model] = rows

    return curves


def save_pr_csv(curves, out_dir):
    import csv

    for model, rows in curves.items():
        path = os.path.join(out_dir, f"{model}_pr.csv")
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["threshold", "precision", "recall", "f1"])
            for r in rows:
                writer.writerow([r["threshold"], r["precision"], r["recall"], r["f1"]])


def plot_single(curves, model, out_dir):
    xs = [r["recall"] for r in curves[model]]
    ys = [r["precision"] for r in curves[model]]

    plt.figure(figsize=(6, 6))
    plt.plot(xs, ys, marker="o", label=model.upper())
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR Curve — {model.upper()}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"pr_{model}.png"))
    plt.close()


def plot_all(curves, out_dir):
    plt.figure(figsize=(8, 6))

    for model, color in zip(["ssd", "yolo", "ensemble"],
                            ["red", "green", "blue"]):
        xs = [r["recall"] for r in curves[model]]
        ys = [r["precision"] for r in curves[model]]
        plt.plot(xs, ys, marker="o", color=color, label=model.upper())

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR Curves — SSD vs YOLO vs Ensemble")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pr_all.png"))
    plt.close()


def main():
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    detector = Detector()

    images_dir = "data/test_images/images"
    labels_dir = "data/test_images/labels"

    out_dir = "data/results/pr_curves"
    os.makedirs(out_dir, exist_ok=True)

    # глобальна структура для накопичення статистики
    global_stats = {model: {thr: {"tp": 0, "fp": 0, "fn": 0}
                            for thr in thresholds}
                    for model in ["ssd", "yolo", "ensemble"]}

    # === проходимо один раз по зображеннях ===
    for filename in os.listdir(images_dir):
        if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img_path = os.path.join(images_dir, filename)
        label_path = os.path.join(labels_dir, filename.replace(".jpg", ".txt"))

        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]
        gt_boxes = load_yolo_label(label_path, w, h)

        preds_all = detector.predict(img, mode="ensemble")

        # Локальна статистика по одному зображенню
        stats = evaluate_model(preds_all, gt_boxes, thresholds)

        # додаємо до глобальної
        for model in ["ssd", "yolo", "ensemble"]:
            for thr in thresholds:
                for key in ["tp", "fp", "fn"]:
                    global_stats[model][thr][key] += stats[model][thr][key]

    # === перетворюємо TP/FP/FN → PR/F1 ===
    curves = compute_pr_curves(global_stats, thresholds)

    # === зберігаємо CSV ===
    save_pr_csv(curves, out_dir)

    # === окремі графіки ===
    for model in ["ssd", "yolo", "ensemble"]:
        plot_single(curves, model, out_dir)

    # === загальна PR-крива ===
    plot_all(curves, out_dir)

    print("PR curves saved to:", out_dir)


if __name__ == "__main__":
    main()
