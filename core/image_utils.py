from __future__ import annotations

import cv2
import numpy as np


class ImageUtils:
    @staticmethod
    def draw_boxes(img, boxes, color=(0, 255, 0), label="model", thickness=2):
        out = img.copy()
        for (x1, y1, x2, y2, score) in boxes:
            cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
            cv2.putText(
                out,
                f"{label}: {score:.2f}",
                (int(x1), int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )
        return out

    @staticmethod
    def draw_multiple_models(img, predictions, thickness=2):
        colors = {
            "ssd": (255, 0, 0),
            "yolo": (0, 255, 0),
            "ensemble": (0, 0, 255),
        }
        out = img.copy()
        for model, boxes in predictions.items():
            color = colors.get(model, (128, 128, 128))
            for (x1, y1, x2, y2, score) in boxes:
                cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
                cv2.putText(
                    out,
                    f"{model}: {score:.2f}",
                    (int(x1), int(y1) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )
        return out

    @staticmethod
    def compute_iou(box_a, box_b):
        """
        box: [x1, y1, x2, y2]
        """
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])

        inter_w = max(0, x2 - x1)
        inter_h = max(0, y2 - y1)
        inter_area = inter_w * inter_h

        area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
        area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
        union = area_a + area_b - inter_area
        if union == 0:
            return 0.0
        return inter_area / union
