import os
import sys
import json
import csv

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

DATA_DIR = "data"
RESULTS_DIR = os.path.join(DATA_DIR, "results")
OUT_DIR = os.path.join(RESULTS_DIR, "compare_models")

os.makedirs(OUT_DIR, exist_ok=True)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_metrics():
    metrics = {}
    for model in ["ssd", "yolo", "ensemble"]:
        path = os.path.join(RESULTS_DIR, model, "metrics.json")
        metrics[model] = load_json(path)
    return metrics


def load_speed():
    speed_path = os.path.join(RESULTS_DIR, "speed", "speed_results.json")
    with open(speed_path, "r") as f:
        data = json.load(f)

    speed = {row["model"]: row for row in data}
    return speed


def save_csv_table(table, out_path):
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(table)


def main():
    print("Loading metrics...")

    metrics = load_metrics()
    speed = load_speed()

    header = [
        "Model",
        "Precision",
        "Recall",
        "F1",
        "IoU_mean",
        "mAP50",
        "TP",
        "FP",
        "FN",
        "FPS",
        "Total_time_sec"
    ]

    rows = [header]

    for m in ["ssd", "yolo", "ensemble"]:
        mt = metrics[m]
        sp = speed[m]

        rows.append([
            m.upper(),
            mt["Precision"],
            mt["Recall"],
            mt["F1"],
            mt["IoU_mean"],
            mt["mAP50"],
            mt["TP"],
            mt["FP"],
            mt["FN"],
            sp["fps"],
            sp["total_time_sec"]
        ])

    csv_path = os.path.join(OUT_DIR, "models_comparison.csv")
    save_csv_table(rows, csv_path)
    print("✓ Saved CSV:", csv_path)

    summary = {
        "ssd": {
            **metrics["ssd"],
            **{"fps": speed["ssd"]["fps"]},
            **{"total_time": speed["ssd"]["total_time_sec"]}
        },
        "yolo": {
            **metrics["yolo"],
            **{"fps": speed["yolo"]["fps"]},
            **{"total_time": speed["yolo"]["total_time_sec"]}
        },
        "ensemble": {
            **metrics["ensemble"],
            **{"fps": speed["ensemble"]["fps"]},
            **{"total_time": speed["ensemble"]["total_time_sec"]}
        }
    }

    json_path = os.path.join(OUT_DIR, "models_comparison.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=4)

    print("✓ Saved JSON:", json_path)

    print("\n🎉 Model comparison completed!")
    print(f"Files saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
