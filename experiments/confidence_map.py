import os
import sys
import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from core.detector import Detector


def boxes_to_heatmap(img, boxes, min_conf=0.05, global_softening=0.3):
    h, w = img.shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float32)

    for b in boxes:
        if len(b) < 5:
            continue

        x1, y1, x2, y2, s = b

        # Враховуємо навіть дуже слабкі
        if s < min_conf:
            s = s * global_softening   # слабі бокси = зменшений вклад

        # Центр бокса
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # Розмір плями = відносно розміру бокса
        bw = max(3, int((abs(x2 - x1)) / 2))
        bh = max(3, int((abs(y2 - y1)) / 2))

        # Gaussian kernel
        xs = np.arange(w)
        ys = np.arange(h)
        xs, ys = np.meshgrid(xs, ys)

        gauss = np.exp(-(((xs - cx) ** 2) / (2 * bw**2) +
                         ((ys - cy) ** 2) / (2 * bh**2)))

        # додаємо вклад (сильні — яскравіші)
        heatmap += gauss * float(s)

    # нормалізація
    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    # згладжування — робить карту "аттеншн мап" і красивою
    heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)

    # візуалізація
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.6, color, 0.4, 0)

    return overlay



def main():
    out_root = "data/results/confidence"
    os.makedirs(os.path.join(out_root, "ssd"), exist_ok=True)
    os.makedirs(os.path.join(out_root, "yolo"), exist_ok=True)
    os.makedirs(os.path.join(out_root, "ensemble"), exist_ok=True)

    images_dir = "data/test_images/images"
    detector = Detector(0.05, 0.05)

    for fname in os.listdir(images_dir)[:10]:
        if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img_path = os.path.join(images_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            continue

        # один прогін – отримуємо всі три набори бокcів
        preds = detector.predict(img, mode="ensemble")
        ssd_boxes = preds["ssd"]
        yolo_boxes = preds["yolo"]
        ens_boxes = preds["ensemble"]

        # SSD map
        ssd_map = boxes_to_heatmap(img, ssd_boxes)
        cv2.imwrite(os.path.join(out_root, "ssd", fname), ssd_map)

        # YOLO map
        yolo_map = boxes_to_heatmap(img, yolo_boxes)
        cv2.imwrite(os.path.join(out_root, "yolo", fname), yolo_map)

        # Ensemble map
        ens_map = boxes_to_heatmap(img, ens_boxes)
        cv2.imwrite(os.path.join(out_root, "ensemble", fname), ens_map)

    print("Confidence maps generated in data/results/confidence/")


if __name__ == "__main__":
    main()
