import os
import sys
import time
import json
import csv
import cv2

# Add project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from core.detector import Detector


def measure_speed(detector, model_name, images_dir):
    """
    Міряє час для передбачення num_images.
    Рахує total_time та FPS = num_images / total_time.
    """

    # Беремо перші num_images файлів
    image_paths = []
    for fname in os.listdir(images_dir):
        if fname.lower().endswith((".jpg", ".png", ".jpeg")):
            image_paths.append(os.path.join(images_dir, fname))


    if len(image_paths) == 0:
        raise RuntimeError("Немає тестових зображень!")

    # Запуск
    start = time.time()

    for img_path in image_paths:
        img = cv2.imread(img_path)
        detector.predict(img, mode=model_name)  # повний inference

    end = time.time()

    total_time = end - start
    num_images = len(image_paths)
    fps = num_images / total_time

    return {
        "model": model_name,
        "num_images": num_images,
        "total_time_sec": total_time,
        "fps": fps
    }


def main():
    out_dir = "data/results/speed"
    os.makedirs(out_dir, exist_ok=True)
    
    detector = Detector()
    images_dir = "data/test_images/images"

    results = []

    for model in ["ssd", "yolo", "ensemble"]:
        print(f"Testing {model.upper()}...")
        res = measure_speed(detector, model, images_dir)
        results.append(res)

    # JSON
    with open(os.path.join(out_dir, "speed_results.json"), "w") as f:
        json.dump(results, f, indent=4)

    # CSV
    with open(os.path.join(out_dir, "speed_results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "num_images", "total_time_sec", "fps"])
        for r in results:
            w.writerow([r["model"], r["num_images"], r["total_time_sec"], r["fps"]])

    print("\n=== SPEED TEST COMPLETE ===")
    print(json.dumps(results, indent=4))


if __name__ == "__main__":
    main()
