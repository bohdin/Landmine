from core.ssd_inference import SSDModel
from core.yolo_inference import YOLO_ONNX
from core.ensemble_inference import EnsembleWBF
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


class Detector:
    """Єдиний інтерфейс над SSD, YOLO та ансамблем."""

    def __init__(self, ssd_conf = 0.4, yolo_conf = 0.4, wbf_iou: float = 0.5, wbf_skip: float = 0.4, wbf_weights=None):
        print("Loading SSD...")
        ssd_path = BASE_DIR / "models" / "ssd300.pth"
        self.ssd = SSDModel(str(ssd_path), device="cpu", conf_threshold= ssd_conf)

        print("Loading YOLO...")
        yolo_path = BASE_DIR / "models" / "yolo.onnx"
        self.yolo = YOLO_ONNX(str(yolo_path), conf_threshold = yolo_conf)
        self.ensemble = EnsembleWBF(iou_thr=wbf_iou, skip_box_thr=wbf_skip, weights=wbf_weights)

    def predict(self, img, mode="ensemble"):
        """Повертає бокси у форматі {model_name: [[x1, y1, x2, y2, score], ...]}."""
        orig_h, orig_w = img.shape[:2]

        ssd_boxes = self.ssd.predict(img) if mode in ("ssd", "ensemble", "all") else []
        yolo_boxes = self.yolo.predict(img) if mode in ("yolo", "ensemble", "all") else []
        preds = {}
        if mode in ("ssd", "ensemble", "all"):
            preds["ssd"] = ssd_boxes
        if mode in ("yolo", "ensemble", "all"):
            preds["yolo"] = yolo_boxes
        if mode in ("ensemble", "all"):
            preds["ensemble"] = self.ensemble.predict(ssd_boxes, yolo_boxes, orig_w, orig_h)
        return preds


if __name__ == "__main__":
    import cv2

    img = cv2.imread("E:\\Code\\Mag_diploma\\Landmine\\data\\test_images\\images\\5_Zone_1_Mine_1cm_depth__2-2m_TemperatureInCenter__33-C_jpg.rf.94dcca373102acf10657071ac661716d.jpg")
    det = Detector()

    print(det.predict(img, mode="ssd"))
    print(det.predict(img, mode="yolo"))
    print(det.predict(img, mode="ensemble"))

    preds = det.predict(img, mode="ensemble")
    img_copy = img.copy()

    x1, y1, x2, y2, _ = preds['ensemble'][0]
    cv2.rectangle(img_copy, (int(x1),int(y1)), (int(x2),int(y2)), (0,0,255), 2)

    cv2.imwrite("ensemble.jpg", img_copy)

