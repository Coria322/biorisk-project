from ultralytics import YOLO
import numpy as np
from app.schemas.detection import DetectionResult, BoundingBox

BIORISK = 'backend/models/modelo_ds_completo.pt'


class YOLOService:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

        self.MIN_CONF = 0.35
        self.MAX_DETECTIONS = 50

        self.ALLOWED_CLASSES = None  # opcional

    def detect(self, frame: np.ndarray, confidence: float = 0.5) -> list[DetectionResult]:

        conf = max(confidence, self.MIN_CONF)

        results = self.model(frame, conf=conf)

        detections = []

        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:

                cls_id = int(box.cls[0])
                label = self.model.names[cls_id]
                score = float(box.conf[0])

                if score < conf:
                    continue

                if self.ALLOWED_CLASSES and label not in self.ALLOWED_CLASSES:
                    continue

                detections.append(
                    DetectionResult(
                        label=label,
                        confidence=round(score, 4),
                        bbox=BoundingBox(
                            x1=int(box.xyxy[0][0]),
                            y1=int(box.xyxy[0][1]),
                            x2=int(box.xyxy[0][2]),
                            y2=int(box.xyxy[0][3])
                        )
                    )
                )

        # ordenar por confianza
        detections.sort(key=lambda d: d.confidence, reverse=True)

        return detections[:self.MAX_DETECTIONS]

    def get_classes(self) -> dict:
        return self.model.names


yolo_service = YOLOService(BIORISK)