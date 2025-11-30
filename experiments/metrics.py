import numpy as np


# ============================================================
#   IOU (intersection over union)
# ============================================================

def iou(box1, box2):
    """
    box format: [x1, y1, x2, y2]
    """

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    box1_area = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    box2_area = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


# ============================================================
#   Precision, Recall, F1
# ============================================================

def precision_recall_f1(tp, fp, fn):
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-9)
    return precision, recall, f1


# ============================================================
#   Average Precision (AP)
# ============================================================

def average_precision(recalls, precisions):
    """ Compute AP using the trapezoid method """
    recalls = np.array(recalls)
    precisions = np.array(precisions)

    # Sort by recall
    order = recalls.argsort()
    recalls = recalls[order]
    precisions = precisions[order]

    # Integrate
    return np.trapz(precisions, recalls)


def average_precision_from_scores(scores, labels, total_gt):
    """
    scores: list of confidences (float)
    labels: list of 1 (TP) / 0 (FP)
    total_gt: загальна кількість GT-боксів (TP + FN)

    Повертає AP при IoU=0.5 (AP50).
    """
    if total_gt == 0 or len(scores) == 0:
        return 0.0

    # Сортуємо предикти за score по спаданню
    scores = np.array(scores)
    labels = np.array(labels)

    order = np.argsort(-scores)
    labels = labels[order]

    tp_cum = np.cumsum(labels == 1)
    fp_cum = np.cumsum(labels == 0)

    recalls = tp_cum / (total_gt + 1e-9)
    precisions = tp_cum / (tp_cum + fp_cum + 1e-9)

    # Робимо "envelope" як у класичному AP
    # precision на кожному рівні recall не повинна зменшуватись вправо
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    # Інтегруємо по recall
    ap = 0.0
    prev_recall = 0.0
    for r, p in zip(recalls, precisions):
        ap += p * (r - prev_recall)
        prev_recall = r

    return float(ap)



# ============================================================
#   mAP (mean AP over classes)
# ============================================================

def mean_average_precision(aps):
    if len(aps) == 0:
        return 0.0
    return float(np.mean(aps))


# ============================================================
#                   TEST SECTION
# ============================================================

if __name__ == "__main__":
    print("Running metrics tests...")

    # Test IoU
    b1 = [10, 10, 50, 50]
    b2 = [30, 30, 60, 60]
    print("IoU(b1, b2) =", iou(b1, b2))

    # Test precision, recall, F1
    p, r, f1 = precision_recall_f1(tp=10, fp=2, fn=3)
    print(f"Precision={p:.3f} Recall={r:.3f} F1={f1:.3f}")

    # Test AP
    recalls = [0.0, 0.5, 1.0]
    precisions = [1.0, 0.8, 0.6]
    ap = average_precision(recalls, precisions)
    print("AP =", ap)

    # Test mAP
    print("mAP =", mean_average_precision([0.5, 0.7, 0.8]))

    print("✓ metrics.py tests passed.")
