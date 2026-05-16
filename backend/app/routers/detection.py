from fastapi import APIRouter, File, UploadFile, Form
from fastapi.responses import StreamingResponse, Response, FileResponse
from starlette.background import BackgroundTask
from app.services.yolo_service import yolo_service
from app.utils.image import decode_image
from app.schemas.detection import DetectionResponse
from app.utils.report import generate_pdf_report
import io, cv2, tempfile, os, shutil

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

@router.post("/video")
async def detect_video(
    file: UploadFile = File(...),
    confidence: float = Form(0.5)
):
    # Crear archivos temporales
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    shutil.copyfileobj(file.file, temp_input)
    input_path = temp_input.name
    temp_input.close()

    output_path = input_path.replace(".mp4", "_out.mp4")

    # Procesar video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return Response(content="Error al abrir el video", status_code=400)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Usar mp4v para máxima compatibilidad con OpenCV
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Inferencia y anotación
        results = yolo_service.model(frame, conf=confidence)[0]
        annotated_frame = results.plot()
        out.write(annotated_frame)

    cap.release()
    out.release()

    # Limpiar entrada
    if os.path.exists(input_path):
        os.remove(input_path)

    # Función para limpiar el archivo de salida después de enviarlo
    def cleanup():
        if os.path.exists(output_path):
            os.remove(output_path)

    return FileResponse(
        output_path, 
        media_type="video/mp4", 
        background=BackgroundTask(cleanup)
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

@router.post("/report", response_class=Response)
async def generate_report(
    file: UploadFile = File(...),
    confidence: float = Form(0.5)
):
    frame = await decode_image(file)

    # Inferencia para anotación visual
    results = yolo_service.model(
        frame,
        conf=confidence,
        iou=0.5,
        agnostic_nms=True
    )[0]

    # Filtrado de boxes
    if results.boxes is not None:
        keep = results.boxes.conf >= confidence
        results.boxes = results.boxes[keep]

    # Extraer lista de detecciones para la tabla
    detections = yolo_service.detect(frame, confidence)

    # Imagen anotada
    annotated = results.plot()
    _, buffer = cv2.imencode(".jpg", annotated)
    annotated_bytes = buffer.tobytes()

    # Generar PDF
    pdf_bytes = generate_pdf_report(annotated_bytes, detections)

    return Response(
        content=pdf_bytes, 
        media_type="application/pdf", 
        headers={"Content-Disposition": "attachment; filename=reporte_biorisk.pdf"}
    )