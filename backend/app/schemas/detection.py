from pydantic import BaseModel

class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int

class DetectionResult(BaseModel):
    label: str
    confidence: str
    bbox: BoundingBox

class DetectionResponse(BaseModel):
    total: int
    image_shape: dict
    detections: list[DetectionResult]
    
class ClassesResponse(BaseModel):
    classes: dict