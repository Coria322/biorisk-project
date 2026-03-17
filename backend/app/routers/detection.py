from fastapi import APIRouter, File, UploadFile, Query
from app.services.yolo_service import yolo_service
from app.utils.image import decode_image
from app.schemas.detection import DetectionResponse

router = APIRouter(prefix="/detect", tags=["Detection"])

@router.post("/", response_model=DetectionResponse)
async def detect(
    file: UploadFile = File(...),
    confidence: float = Query(default=0.5, ge=0.0, le=1.0)
):
    frame = await decode_image(file)
    detections = yolo_service.detect(frame, confidence)

    return DetectionResponse(
        total=len(detections),
        image_shape={"height": frame.shape[0], "width": frame.shape[1]},
        detections=detections
    )

@router.post("/image", response_model=None)
async def detect_image(file: UploadFile = File(...)):
    from fastapi.responses import StreamingResponse
    import io, cv2

    frame = await decode_image(file)
    results = yolo_service.model(frame)
    annotated = results[0].plot()

    _, buffer = cv2.imencode(".jpg", annotated)
    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")

