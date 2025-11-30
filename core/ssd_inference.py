import torch
import cv2
import numpy as np

import torch
from torchvision.models.detection import ssd300_vgg16
from torchvision.models.detection.ssd import SSDHead
from torchvision.models.detection import _utils as det_utils


class SSDModel:
    def __init__(self, weight_path="models/ssd300.pth", device="cpu", conf_threshold=0.4):
        self.device = device
        self.conf_threshold = conf_threshold
        self.input_size = 480

        # Create SSD arch
        self.model = ssd300_vgg16(weights=None)

        # Rebuild HEAD for 1-class detection
        in_channels = det_utils.retrieve_out_channels(self.model.backbone, (self.input_size, self.input_size))
        num_anchors = self.model.anchor_generator.num_anchors_per_location()

        self.model.head = SSDHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
            num_classes=2
        )

        # Load weights
        state_dict = torch.load(weight_path, map_location=device)
        self.model.load_state_dict(state_dict)

        self.model.to(device)
        self.model.eval()

    def preprocess(self, img):
        img_resized = cv2.resize(img, (self.input_size, self.input_size))
        img_resized = img_resized.astype(np.float32) / 255.0
        img_resized = img_resized.transpose(2, 0, 1)
        tensor = torch.tensor(img_resized).unsqueeze(0)
        return tensor

    def predict(self, img):
        orig_h, orig_w = img.shape[:2]

        # preprocess
        tensor = self.preprocess(img).to(self.device)

        # predict
        with torch.no_grad():
            out = self.model(tensor)[0]

        boxes = out["boxes"].cpu().numpy()
        scores = out["scores"].cpu().numpy()

        results = []
        for b, s in zip(boxes, scores):
            if s >= self.conf_threshold:
                x1 = b[0] / self.input_size * orig_w
                y1 = b[1] / self.input_size * orig_h
                x2 = b[2] / self.input_size * orig_w
                y2 = b[3] / self.input_size * orig_h

                results.append([
                    float(x1), float(y1), float(x2), float(y2), float(s)
                ])

        return results



if __name__ == "__main__":
    model = SSDModel('E:\\Code\\Mag_diploma\\Landmine\\models\\ssd300.pth')

    img = cv2.imread('E:\\Code\\Mag_diploma\\Landmine\\data\\test_images\\images\\5_Zone_1_Mine_1cm_depth__2-2m_TemperatureInCenter__33-C_jpg.rf.94dcca373102acf10657071ac661716d.jpg')

    preds = model.predict(img)
    print("SSD predictions:", preds)

    img_copy = img.copy()


    # YOLO box — green
    x1, y1, x2, y2, _ = preds[0]
    cv2.rectangle(img_copy, (int(x1),int(y1)), (int(x2),int(y2)), (255,0,0), 2)

    cv2.imwrite("ssd300.jpg", img_copy)
