from fastapi import APIRouter, File, UploadFile, Form
from fastapi.responses import StreamingResponse
from app.services.yolo_service import yolo_service
from app.utils.image import decode_image
from app.schemas.detection import DetectionResponse
import io, cv2

router = APIRouter(prefix="/detect", tags=["Detection"])


@router.post("/", response_model=DetectionResponse)
async def detect(
    file: UploadFile = File(...),
    confidence: float = Form(0.5)
):
    frame = await decode_image(file)

    detections = yolo_service.detect(frame, confidence)

    return DetectionResponse(
        total=len(detections),
        image_shape={"height": frame.shape[0], "width": frame.shape[1]},
        detections=detections
    )


@router.post("/image", response_class=StreamingResponse)
async def detect_image(
    file: UploadFile = File(...),
    confidence: float = Form(0.5)
):
    frame = await decode_image(file)

    # 🔥 inference con filtros correctos desde YOLO
    results = yolo_service.model(
        frame,
        conf=confidence,
        iou=0.5,
        agnostic_nms=True
    )[0]

    # 🔥 FILTRADO SEGURO (NO modificar boxes manualmente)
    if results.boxes is not None:
        keep = results.boxes.conf >= confidence
        results.boxes = results.boxes[keep]

    annotated = results.plot()

    _, buffer = cv2.imencode(".jpg", annotated)
    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")