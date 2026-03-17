from ultralytics import YOLO
import numpy as np
from app.schemas.detection import DetectionResult, BoundingBox

class YOLOService:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def detect(self, frame: np.ndarray, confidence: float = 0.5) -> list[DetectionResult]:
        results = self.model(frame, conf=confidence)
        detections = []

        for r in results:
            for box in r.boxes:
                detections.append(DetectionResult(
                    label=self.model.names[int(box.cls[0])],
                    confidence=round(float(box.conf[0]), 4),
                    bbox=BoundingBox(
                        x1=int(box.xyxy[0][0]),
                        y1=int(box.xyxy[0][1]),
                        x2=int(box.xyxy[0][2]),
                        y2=int(box.xyxy[0][3])
                    )
                ))

        return detections

    def get_classes(self) -> dict:
        return self.model.names

yolo_service = YOLOService("backend/models/yolov8n.pt")