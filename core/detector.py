from ssd_inference import SSDModel
from yolo_inference import YOLO_ONNX
from ensemble_inference import ensemble_boxes_custom


class Detector:
    def __init__(self):
        print("Loading SSD...")
        self.ssd = SSDModel("E:\\Code\\Mag_diploma\\Landmine\\models\\ssd300.pth", device="cpu")

        print("Loading YOLO...")
        self.yolo = YOLO_ONNX("E:\\Code\\Mag_diploma\\Landmine\\models\\yolo.onnx")

    def predict(self, img, mode="ensemble"):
        orig_h, orig_w = img.shape[:2]

        # SSD
        ssd_boxes = self.ssd.predict(img) if mode in ("ssd", "ensemble") else []

        # YOLO
        yolo_boxes = self.yolo.predict(img) if mode in ("yolo", "ensemble") else []

        # Ensemble
        if mode == "ensemble":
            fused = ensemble_boxes_custom(ssd_boxes, yolo_boxes, orig_w, orig_h)
            return {"ssd": ssd_boxes, "yolo": yolo_boxes, "ensemble": fused}

        if mode == "ssd":
            return {"ssd": ssd_boxes}

        if mode == "yolo":
            return {"yolo": yolo_boxes}


if __name__ == "__main__":
    import cv2

    img = cv2.imread("E:\\Code\\Mag_diploma\\Landmine\\data\\test_images\\images\\5_Zone_1_Mine_1cm_depth__2-2m_TemperatureInCenter__33-C_jpg.rf.94dcca373102acf10657071ac661716d.jpg")
    det = Detector()

    print(det.predict(img, mode="ssd"))
    print(det.predict(img, mode="yolo"))
    print(det.predict(img, mode="ensemble"))

    preds = det.predict(img, mode="ensemble")
    img_copy = img.copy()


    # YOLO box — green
    x1, y1, x2, y2, _ = preds['ensemble'][0]
    cv2.rectangle(img_copy, (int(x1),int(y1)), (int(x2),int(y2)), (0,0,255), 2)

    cv2.imwrite("ensemble.jpg", img_copy)

