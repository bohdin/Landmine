import os
import sys
import json
import csv
import math

import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)


DATA_DIR = os.path.join("data")
RESULTS_DIR = os.path.join(DATA_DIR, "results")
LOSS_DIR = os.path.join(DATA_DIR, "loss_history")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")



def ensure_dirs():
    os.makedirs(PLOTS_DIR, exist_ok=True)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_csv_dicts(path):
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)



def load_metrics():
    metrics = {}
    for model in ["ssd", "yolo", "ensemble"]:
        path = os.path.join(RESULTS_DIR, model, "metrics.json")
        metrics[model] = load_json(path)
    return metrics


def load_speed():
    path = os.path.join(RESULTS_DIR, "speed", "speed_results.json")
    data = load_json(path)
    speed = {row["model"]: row for row in data}
    return speed


def load_pr_curves():
    pr = {}
    for model in ["ssd", "yolo", "ensemble"]:
        path = os.path.join(RESULTS_DIR, "pr_curves", f"{model}_pr.csv")
        rows = load_csv_dicts(path)
        thresholds = [float(r["threshold"]) for r in rows]
        precisions = [float(r["precision"]) for r in rows]
        recalls = [float(r["recall"]) for r in rows]
        pr[model] = {"threshold": thresholds,
                     "precision": precisions,
                     "recall": recalls}
    return pr



def plot_radar(metrics, speed):
    labels = ["Precision", "Recall", "F1", "IoU_mean", "mAP50", "FPS"]

    max_fps = max(speed[m]["fps"] for m in ["ssd", "yolo", "ensemble"])

    def norm_vals(m):
        return [
            metrics[m]["Precision"],
            metrics[m]["Recall"],
            metrics[m]["F1"],
            metrics[m]["IoU_mean"],
            metrics[m]["mAP50"],
            speed[m]["fps"] / max_fps,
        ]

    models = ["ssd", "yolo", "ensemble"]
    color_map = {"ssd": "tab:blue", "yolo": "tab:green", "ensemble": "tab:red"}

    num_vars = len(labels)
    angles = np.linspace(0, 2 * math.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)

    for m in models:
        v = norm_vals(m)
        v += v[:1]
        ax.plot(angles, v, label=m.upper(), color=color_map[m], linewidth=2)
        ax.fill(angles, v, alpha=0.15, color=color_map[m])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_rlabel_position(0)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
    plt.title("Model comparison (Normalized Radar chart)")

    out_path = os.path.join(PLOTS_DIR, "radar.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Збережено: {out_path}")



def plot_grouped_bars(metrics, speed):
    models = ["ssd", "yolo", "ensemble"]
    metrics_labels = ["Precision", "Recall", "F1", "IoU_mean", "mAP50", "FPS"]

    data = {m: [] for m in models}
    for m in models:
        data[m].append(metrics[m]["Precision"])
        data[m].append(metrics[m]["Recall"])
        data[m].append(metrics[m]["F1"])
        data[m].append(metrics[m]["IoU_mean"])
        data[m].append(metrics[m]["mAP50"])
        data[m].append(speed[m]["fps"])

    x = np.arange(len(metrics_labels))
    width = 0.25

    plt.figure(figsize=(10, 6))
    plt.bar(x - width, data["ssd"], width, label="SSD")
    plt.bar(x, data["yolo"], width, label="YOLO")
    plt.bar(x + width, data["ensemble"], width, label="Ensemble")

    plt.xticks(x, metrics_labels)
    plt.ylabel("Value")
    plt.title("Models metrics comparison (bar chart)")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.3)

    out_path = os.path.join(PLOTS_DIR, "metrics_bars.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Збережено: {out_path}")



def plot_pr_curves(pr_curves):
    plt.figure(figsize=(8, 6))
    colors = {"ssd": "tab:blue", "yolo": "tab:green", "ensemble": "tab:red"}

    for model in ["ssd", "yolo", "ensemble"]:
        pr = pr_curves[model]
        plt.plot(pr["recall"], pr["precision"],
                 marker="o", label=model.upper(), color=colors[model])

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curves")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()

    out_path = os.path.join(PLOTS_DIR, "pr_curves.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Збережено: {out_path}")



def plot_speed_vs_accuracy(metrics, speed):
    plt.figure(figsize=(8, 6))

    for m, color in zip(["ssd", "yolo", "ensemble"],
                        ["tab:blue", "tab:green", "tab:red"]):
        fps = speed[m]["fps"]
        map50 = metrics[m]["mAP50"]
        plt.scatter(fps, map50, s=80, color=color)
        plt.text(fps * 1.01, map50 * 1.01, m.upper())

    plt.xlabel("FPS (higher is better)")
    plt.ylabel("mAP@0.5 (higher is better)")
    plt.title("Speed vs Accuracy tradeoff")
    plt.grid(True, linestyle="--", alpha=0.4)

    out_path = os.path.join(PLOTS_DIR, "speed_vs_accuracy.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Збережено: {out_path}")



def plot_error_bars(metrics):
    models = ["ssd", "yolo", "ensemble"]
    tps = [metrics[m]["TP"] for m in models]
    fps = [metrics[m]["FP"] for m in models]
    fns = [metrics[m]["FN"] for m in models]

    x = np.arange(len(models))
    width = 0.25

    plt.figure(figsize=(8, 6))
    plt.bar(x - width, tps, width, label="TP", color="tab:green")
    plt.bar(x, fps, width, label="FP", color="tab:orange")
    plt.bar(x + width, fns, width, label="FN", color="tab:red")

    plt.xticks(x, [m.upper() for m in models])
    plt.ylabel("Count")
    plt.title("TP / FP / FN comparison")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.3)

    out_path = os.path.join(PLOTS_DIR, "errors.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Збережено: {out_path}")



def plot_ssd_loss():
    path = os.path.join(LOSS_DIR, "sdd300_loss_history.json")
    hist = load_json(path)

    train = hist.get("training_loss", [])
    val = hist.get("validation_loss", [])

    epochs = range(1, len(train) + 1)

    plt.figure(figsize=(8, 6))
    if train:
        plt.plot(epochs, train, label="Train loss")
    if val and len(val) == len(train):
        plt.plot(epochs, val, label="Val loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SSD300 training history")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()

    out_path = os.path.join(PLOTS_DIR, "loss_ssd.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Збережено: {out_path}")


def plot_yolo_total_loss_with_val():
    path = os.path.join(LOSS_DIR, "yolo_history.csv")
    rows = load_csv_dicts(path)

    if not rows:
        return

    epochs = [int(r.get("epoch", i + 1)) for i, r in enumerate(rows)]

    train_total_loss = []
    val_total_loss = []

    for r in rows:
        try:
            box = float(r.get("train/box_loss", 0))
            cls = float(r.get("train/cls_loss", 0))
            dfl = float(r.get("train/dfl_loss", 0))
            train_total_loss.append(box + cls + dfl)
        except ValueError:
            train_total_loss.append(0)

        try:
            vbox = float(r.get("val/box_loss", 0))
            vcls = float(r.get("val/cls_loss", 0))
            vdfl = float(r.get("val/dfl_loss", 0))
            val_total_loss.append(vbox + vcls + vdfl)
        except ValueError:
            val_total_loss.append(0)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_total_loss, label="Train Total Loss", color="red")
    plt.plot(epochs, val_total_loss, label="Validation Total Loss", color="blue")

    plt.xlabel("Epoch")
    plt.ylabel("Total Loss")
    plt.title("YOLO Total Loss (Train + Validation)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()

    out_path = os.path.join(PLOTS_DIR, "total_loss_yolo.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Збережено: {out_path}")



def main():
    ensure_dirs()

    metrics = load_metrics()
    speed = load_speed()
    pr_curves = load_pr_curves()

    plot_radar(metrics, speed)
    plot_grouped_bars(metrics, speed)
    plot_pr_curves(pr_curves)
    plot_speed_vs_accuracy(metrics, speed)
    plot_error_bars(metrics)
    plot_ssd_loss()
    plot_yolo_total_loss_with_val()

    print(f"\nУсі графіки збережено в: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
