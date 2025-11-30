from ensemble_boxes import weighted_boxes_fusion


def ensemble_boxes_custom(boxes_ssd, boxes_yolo, orig_w, orig_h,
                          iou_thr=0.5, skip_box_thr=0.4):
    """
    boxes_ssd / boxes_yolo -> list of [x1, y1, x2, y2, score]
    orig_w, orig_h -> original image size
    """

    if len(boxes_ssd) == 0 and len(boxes_yolo) == 0:
        return []

    norm_boxes = []
    norm_scores = []
    norm_labels = []

    # SSD
    for b in boxes_ssd:
        norm_boxes.append([b[0] / orig_w, b[1] / orig_h,
                           b[2] / orig_w, b[3] / orig_h])
        norm_scores.append(b[4])
        norm_labels.append(0)

    # YOLO
    for b in boxes_yolo:
        norm_boxes.append([b[0] / orig_w, b[1] / orig_h,
                           b[2] / orig_w, b[3] / orig_h])
        norm_scores.append(b[4])
        norm_labels.append(0)

    boxes_input = [[b] for b in norm_boxes]
    scores_input = [[s] for s in norm_scores]
    labels_input = [[l] for l in norm_labels]

    fused_boxes, fused_scores, _ = weighted_boxes_fusion(
        boxes_input, scores_input, labels_input,
        iou_thr=iou_thr,
        skip_box_thr=skip_box_thr
    )

    results = []

    for box, score in zip(fused_boxes, fused_scores):
        x1 = box[0] * orig_w
        y1 = box[1] * orig_h
        x2 = box[2] * orig_w
        y2 = box[3] * orig_h
        results.append([float(x1), float(y1), float(x2), float(y2), float(score)])

    return results
