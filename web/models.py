from core.detector import Detector

# Глобальний інстанс, завантажується 1 раз
detector = Detector(
    ssd_conf=0.4,     # базовий поріг
    yolo_conf=0.4
)

def set_thresholds(ssd_thr: float, yolo_thr: float):
    detector.ssd.conf_threshold = ssd_thr
    detector.yolo.conf_threshold = yolo_thr
