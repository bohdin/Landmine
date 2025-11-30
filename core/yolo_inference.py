import cv2
import numpy as np
import onnxruntime as ort


class YOLO_ONNX:
    def __init__(self, model_path="models/yolo.onnx", conf_threshold=0.4, iou_threshold=0.5):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )

        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, img):
        h0, w0 = img.shape[:2]
        img_resized = cv2.resize(img, (640, 640))
        img_norm = img_resized[:, :, ::-1].astype(np.float32) / 255.0
        img_norm = img_norm.transpose(2, 0, 1)
        return np.expand_dims(img_norm, axis=0), (h0, w0)

    def nms(self, boxes, scores, iou_threshold=0.5):
        idxs = np.argsort(scores)[::-1]
        keep = []

        while len(idxs) > 0:
            i = idxs[0]
            keep.append(i)
            if len(idxs) == 1:
                break

            rest = idxs[1:]
            ious = self.iou(boxes[i], boxes[rest])

            idxs = rest[ious < iou_threshold]

        return keep

    def iou(self, box1, boxes):
        x1 = np.maximum(box1[0], boxes[:, 0])
        y1 = np.maximum(box1[1], boxes[:, 1])
        x2 = np.minimum(box1[2], boxes[:, 2])
        y2 = np.minimum(box1[3], boxes[:, 3])

        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        return inter / (area1 + area2 - inter + 1e-6)

    def predict(self, img):
        inp, (h0, w0) = self.preprocess(img)
        out = self.session.run(None, {self.input_name: inp})[0]

        out = np.squeeze(out)
        out = out.transpose(1, 0)

        boxes = []
        scores = []

        for (x, y, w, h, conf) in out:
            if conf < self.conf_threshold:
                continue

            x1 = x - w / 2
            y1 = y - h / 2
            x2 = x + w / 2
            y2 = y + h / 2

            boxes.append([x1, y1, x2, y2])
            scores.append(conf)

        if len(boxes) == 0:
            return []

        boxes = np.array(boxes)
        scores = np.array(scores)

        keep = self.nms(boxes, scores, self.iou_threshold)

        results = []
        for i in keep:
            x1, y1, x2, y2 = boxes[i]

            x1 = x1 / 640 * w0
            y1 = y1 / 640 * h0
            x2 = x2 / 640 * w0
            y2 = y2 / 640 * h0

            results.append([
                float(x1), float(y1), float(x2), float(y2), float(scores[i])
            ])

        return results


if __name__ == "__main__":
    model = YOLO_ONNX(
        model_path=r"E:\Code\Mag_diploma\Landmine\models\yolo.onnx"
    )

    img = cv2.imread(
        r"E:\Code\Mag_diploma\Landmine\data\test_images\images\5_Zone_1_Mine_1cm_depth__2-2m_TemperatureInCenter__33-C_jpg.rf.94dcca373102acf10657071ac661716d.jpg"
    )

    preds = model.predict(img)
    print("YOLO:", preds)

    img_copy = img.copy()


    # YOLO box — green
    x1, y1, x2, y2, _ = preds[0]
    cv2.rectangle(img_copy, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)

    cv2.imwrite("yolo.jpg", img_copy)