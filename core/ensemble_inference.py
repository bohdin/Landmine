from ensemble_boxes import weighted_boxes_fusion


def ensemble_boxes_custom(boxes_ssd, boxes_yolo, orig_w, orig_h,
                          iou_thr=0.5, skip_box_thr=0.4, weights=None):
    """
    Зливає бокси SSD та YOLO через WBF.
    boxes_ssd / boxes_yolo: список [x1, y1, x2, y2, score] у пікселях оригіналу.
    orig_w, orig_h: розмір оригінального зображення.
    weights: опційні ваги моделей (за замовчанням рівні).
    """

    if len(boxes_ssd) == 0 and len(boxes_yolo) == 0:
        return []

    def normalize_boxes(boxes):
        bxs = []
        scores = []
        for b in boxes:
            bxs.append([b[0] / orig_w, b[1] / orig_h,
                        b[2] / orig_w, b[3] / orig_h])
            scores.append(b[4])
        return bxs, scores

    ssd_boxes_norm, ssd_scores = normalize_boxes(boxes_ssd)
    yolo_boxes_norm, yolo_scores = normalize_boxes(boxes_yolo)

    boxes_input = []
    scores_input = []
    labels_input = []

    if ssd_boxes_norm:
        boxes_input.append(ssd_boxes_norm)
        scores_input.append(ssd_scores)
        labels_input.append([0] * len(ssd_boxes_norm))

    if yolo_boxes_norm:
        boxes_input.append(yolo_boxes_norm)
        scores_input.append(yolo_scores)
        labels_input.append([0] * len(yolo_boxes_norm))

    if not boxes_input:
        return []

    model_weights = weights if weights is not None else [1.0] * len(boxes_input)

    fused_boxes, fused_scores, _ = weighted_boxes_fusion(
        boxes_input,
        scores_input,
        labels_input,
        iou_thr=iou_thr,
        skip_box_thr=skip_box_thr,
        weights=model_weights
    )

    results = []

    for box, score in zip(fused_boxes, fused_scores):
        x1 = box[0] * orig_w
        y1 = box[1] * orig_h
        x2 = box[2] * orig_w
        y2 = box[3] * orig_h
        results.append([float(x1), float(y1), float(x2), float(y2), float(score)])

    return results
