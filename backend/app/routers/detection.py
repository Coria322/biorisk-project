from fastapi import APIRouter, File, UploadFile, Form
from fastapi.responses import StreamingResponse, Response, FileResponse
from starlette.background import BackgroundTask
from app.services.yolo_service import yolo_service
from app.utils.image import decode_image
from app.schemas.detection import DetectionResponse, DetectionResult, BoundingBox
from app.utils.report import generate_pdf_report, generate_pdf_report_video
import io, cv2, tempfile, os, shutil, subprocess, json

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

def calculate_iou(box1, box2):
    """
    Calcula la Intersección sobre Unión (IoU) entre dos cajas delimitadoras [x1, y1, x2, y2].
    """
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (box1[2] - box1[0]) * (box1[3] - box1[1])
    boxBArea = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    unionArea = float(boxAArea + boxBArea - interArea)
    if unionArea <= 0.0:
        return 0.0
    return interArea / unionArea

@router.post("/video")
async def detect_video(
    file: UploadFile = File(...),
    confidence: float = Form(0.5)
):
    print("\n=================== INICIO DETECT_VIDEO ===================")
    print(f"[BACKEND] Archivo recibido: {file.filename}, Content-Type: {file.content_type}")
    print(f"[BACKEND] Confianza del slider: {confidence}")

    # Crear archivos temporales
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    shutil.copyfileobj(file.file, temp_input)
    input_path = temp_input.name
    temp_input.close()
    print(f"[BACKEND] Archivo temporal de entrada creado en: {input_path}")

    output_path = input_path.replace(".mp4", "_out.mp4")
    print(f"[BACKEND] Ruta de salida OpenCV planeada: {output_path}")

    # Procesar video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("[BACKEND] [ERROR] ¡No se pudo abrir el archivo de video con OpenCV!")
        return Response(content="Error al abrir el video", status_code=400)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"[BACKEND] Video abierto. Dimensiones: {width}x{height}, FPS: {fps}")

    # Usar mp4v para máxima compatibilidad con OpenCV
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        print("[BACKEND] [ERROR] ¡No se pudo inicializar VideoWriter de OpenCV!")

    aggregated_detections = {}
    frame_idx = 0

    # Umbral bajo para mantenimiento (40% del umbral del slider, min 0.05)
    low_confidence = max(0.05, confidence * 0.4)
    print(f"[BACKEND] [HISTÉRESIS ESPACIAL] Umbral de bloqueo (Lock-on): {confidence}, Umbral de mantenimiento: {low_confidence}")

    # Lista de tracks activos. Cada track: {"bbox": [x1, y1, x2, y2], "label": str, "grace_frames": int, "active": bool}
    active_tracks = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Inferencia con umbral bajo para capturar detecciones candidatas de tracking
        results = yolo_service.model(
            frame,
            conf=low_confidence,
            iou=0.5,
            agnostic_nms=True
        )[0]

        keep_indices = []
        matched_track_indices = set()

        # Filtrado de boxes aplicando histéresis espacial (IoU)
        if results.boxes is not None and len(results.boxes) > 0:
            current_dets = []
            for idx, box in enumerate(results.boxes):
                cls_id = int(box.cls[0])
                label = yolo_service.model.names[cls_id]
                score = float(box.conf[0])
                bbox = box.xyxy[0].tolist() # [x1, y1, x2, y2]
                current_dets.append({
                    "idx": idx,
                    "bbox": bbox,
                    "label": label,
                    "score": score,
                    "box_obj": box,
                    "matched": False
                })
            
            # Ordenamos las detecciones por confianza descendente para dar prioridad a las más fuertes
            current_dets.sort(key=lambda x: x["score"], reverse=True)
            
            # Intentar asociar detecciones con tracks existentes (por clase y proximidad IoU)
            for det in current_dets:
                best_iou = 0.0
                best_track_idx = -1
                
                # Buscar el track del mismo label con mayor IoU
                for t_idx, track in enumerate(active_tracks):
                    if t_idx in matched_track_indices:
                        continue
                    if track["label"] != det["label"]:
                        continue
                    
                    iou = calculate_iou(track["bbox"], det["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_track_idx = t_idx
                
                # Si encontramos un track coincidente (IoU >= 0.25)
                if best_track_idx != -1 and best_iou >= 0.25:
                    track = active_tracks[best_track_idx]
                    matched_track_indices.add(best_track_idx)
                    det["matched"] = True
                    
                    # Actualizar el track con la nueva posición y confianza
                    track["bbox"] = det["bbox"]
                    track["grace_frames"] = 15  # Refrescar rastro/gracia
                    
                    # Si la confianza de la detección es fuerte, activar el track si no lo estaba
                    if det["score"] >= confidence:
                        track["active"] = True
                    
                    # Si el track está activo (locked-on), conservamos esta detección en la salida
                    if track["active"]:
                        keep_indices.append(det["idx"])
                        
                        # Guardar en detecciones agregadas (para el panel derecho del frontend)
                        lbl = det["label"]
                        scr = det["score"]
                        if lbl not in aggregated_detections or scr > aggregated_detections[lbl]["confidence"]:
                            aggregated_detections[lbl] = {
                                "label": lbl,
                                "confidence": round(scr, 4),
                                "bbox": {
                                    "x1": int(det["bbox"][0]),
                                    "y1": int(det["bbox"][1]),
                                    "x2": int(det["bbox"][2]),
                                    "y2": int(det["bbox"][3])
                                }
                            }
            
            # Para las detecciones NO asociadas:
            # Si superan el umbral fuerte de confianza (slider), crean un NUEVO track activo
            for det in current_dets:
                if not det["matched"]:
                    if det["score"] >= confidence:
                        new_track = {
                            "bbox": det["bbox"],
                            "label": det["label"],
                            "grace_frames": 15,
                            "active": True
                        }
                        active_tracks.append(new_track)
                        keep_indices.append(det["idx"])
                        
                        # Guardar en detecciones agregadas
                        lbl = det["label"]
                        scr = det["score"]
                        if lbl not in aggregated_detections or scr > aggregated_detections[lbl]["confidence"]:
                            aggregated_detections[lbl] = {
                                "label": lbl,
                                "confidence": round(scr, 4),
                                "bbox": {
                                    "x1": int(det["bbox"][0]),
                                    "y1": int(det["bbox"][1]),
                                    "x2": int(det["bbox"][2]),
                                    "y2": int(det["bbox"][3])
                                }
                            }
            
            # Actualizar el objeto results para graficar únicamente las cajas seleccionadas
            results.boxes = results.boxes[keep_indices]

        # Decrementar frames de gracia para tracks activos que no se asociaron en este frame
        new_active_tracks = []
        for t_idx, track in enumerate(active_tracks):
            if t_idx not in matched_track_indices:
                track["grace_frames"] -= 1
                if track["grace_frames"] > 0:
                    new_active_tracks.append(track)
                else:
                    print(f"[BACKEND] [HISTÉRESIS] Se eliminó track por inactividad espacial: '{track['label']}'")
            else:
                new_active_tracks.append(track)
        active_tracks = new_active_tracks

        annotated_frame = results.plot()
        out.write(annotated_frame)

    print(f"[BACKEND] Fin de lectura. Total cuadros leídos y procesados: {frame_idx}")
    cap.release()
    out.release()
    print("[BACKEND] Capturas y escritores OpenCV liberados.")

    # Limpiar entrada
    if os.path.exists(input_path):
        os.remove(input_path)
        print("[BACKEND] Archivo temporal de entrada eliminado.")

    # Transcodificar con ffmpeg a H.264 para compatibilidad con navegadores
    temp_ffmpeg_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    ffmpeg_output_path = temp_ffmpeg_out.name
    temp_ffmpeg_out.close()
    print(f"[BACKEND] Iniciando transcodificación ffmpeg a H.264...")
    print(f"[BACKEND] Archivo temporal de destino ffmpeg: {ffmpeg_output_path}")

    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", output_path,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-an",
            ffmpeg_output_path
        ]
        # Capturar salida para imprimirla en caso de problemas
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("[BACKEND] [FFMPEG] Transcodificación completada con éxito.")
        
        # Eliminar el archivo intermedio de OpenCV
        if os.path.exists(output_path):
            os.remove(output_path)
        output_path = ffmpeg_output_path
    except Exception as e:
        print(f"[BACKEND] [ERROR] Falló la transcodificación ffmpeg: {e}")
        if hasattr(e, 'stderr') and e.stderr:
            print(f"[BACKEND] [FFMPEG STDERR] {e.stderr}")
        # Si falla, eliminamos el archivo temporal fallido
        if os.path.exists(ffmpeg_output_path):
            os.remove(ffmpeg_output_path)

    # Función para limpiar el archivo de salida después de enviarlo
    def cleanup():
        if os.path.exists(output_path):
            os.remove(output_path)
            print("[BACKEND] Archivo de salida temporal eliminado de forma segura.")

    # Serializar resultados de detección
    detections_data = {
        "total": len(aggregated_detections),
        "image_shape": {"height": height, "width": width},
        "detections": list(aggregated_detections.values())
    }
    detections_json = json.dumps(detections_data)
    print(f"[BACKEND] Detecciones agregadas enviadas: {detections_json}")
    print("=================== FIN DETECT_VIDEO ===================\n")

    return FileResponse(
        output_path, 
        media_type="video/mp4", 
        headers={"x-detections": detections_json},
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
    # Verificar si es video
    is_video = file.content_type.startswith("video/") or file.filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))

    if is_video:
        print("\n=================== GENERANDO REPORTE DE VIDEO ===================")
        # Crear archivo temporal para leer el video con OpenCV
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        shutil.copyfileobj(file.file, temp_input)
        input_path = temp_input.name
        temp_input.close()
        
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            if os.path.exists(input_path):
                os.remove(input_path)
            return Response(content="Error al abrir el video para reporte", status_code=400)
            
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        sample_interval = max(1, int(fps / 10))  # Muestrear ~10 frames por segundo para velocidad
        
        frames_with_scores = []
        active_tracks = []
        low_confidence = max(0.05, confidence * 0.4)
        aggregated_detections = {}
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_idx += 1
            if frame_idx % sample_interval != 0:
                continue
                
            # Correr inferencia con umbral bajo
            results = yolo_service.model(
                frame,
                conf=low_confidence,
                iou=0.5,
                agnostic_nms=True
            )[0]
            
            keep_indices = []
            matched_track_indices = set()
            max_score = 0.0
            
            if results.boxes is not None and len(results.boxes) > 0:
                current_dets = []
                for idx, box in enumerate(results.boxes):
                    cls_id = int(box.cls[0])
                    label = yolo_service.model.names[cls_id]
                    score = float(box.conf[0])
                    bbox = box.xyxy[0].tolist()
                    current_dets.append({
                        "idx": idx,
                        "bbox": bbox,
                        "label": label,
                        "score": score,
                        "matched": False
                    })
                    
                current_dets.sort(key=lambda x: x["score"], reverse=True)
                
                for det in current_dets:
                    best_iou = 0.0
                    best_track_idx = -1
                    for t_idx, track in enumerate(active_tracks):
                        if t_idx in matched_track_indices:
                            continue
                        if track["label"] != det["label"]:
                            continue
                        iou = calculate_iou(track["bbox"], det["bbox"])
                        if iou > best_iou:
                            best_iou = iou
                            best_track_idx = t_idx
                            
                    if best_track_idx != -1 and best_iou >= 0.25:
                        track = active_tracks[best_track_idx]
                        matched_track_indices.add(best_track_idx)
                        det["matched"] = True
                        track["bbox"] = det["bbox"]
                        track["grace_frames"] = 15
                        if det["score"] >= confidence:
                            track["active"] = True
                        if track["active"]:
                            keep_indices.append(det["idx"])
                            max_score = max(max_score, det["score"])
                            
                            # Guardar en detecciones agregadas de mayor confianza global
                            lbl = det["label"]
                            scr = det["score"]
                            if lbl not in aggregated_detections or scr > aggregated_detections[lbl]["confidence"]:
                                aggregated_detections[lbl] = {
                                    "label": lbl,
                                    "confidence": scr,
                                    "bbox": det["bbox"]
                                }
                                
                # Detecciones no asociadas
                for det in current_dets:
                    if not det["matched"] and det["score"] >= confidence:
                        new_track = {
                            "bbox": det["bbox"],
                            "label": det["label"],
                            "grace_frames": 15,
                            "active": True
                        }
                        active_tracks.append(new_track)
                        keep_indices.append(det["idx"])
                        max_score = max(max_score, det["score"])
                        
                        # Guardar en detecciones agregadas
                        lbl = det["label"]
                        scr = det["score"]
                        if lbl not in aggregated_detections or scr > aggregated_detections[lbl]["confidence"]:
                            aggregated_detections[lbl] = {
                                "label": lbl,
                                "confidence": scr,
                                "bbox": det["bbox"]
                            }
                
                results.boxes = results.boxes[keep_indices]
                
            # Decrementar grace_frames
            new_active_tracks = []
            for t_idx, track in enumerate(active_tracks):
                if t_idx not in matched_track_indices:
                    track["grace_frames"] -= 1
                    if track["grace_frames"] > 0:
                        new_active_tracks.append(track)
                else:
                    new_active_tracks.append(track)
            active_tracks = new_active_tracks
            
            # Si este frame tiene detecciones válidas, lo guardamos
            if len(keep_indices) > 0:
                annotated_frame = results.plot()
                _, buffer = cv2.imencode(".jpg", annotated_frame)
                frame_bytes = buffer.tobytes()
                
                frames_with_scores.append({
                    "max_score": max_score,
                    "frame_bytes": frame_bytes
                })
                
        cap.release()
        if os.path.exists(input_path):
            os.remove(input_path)
            
        # Seleccionar los 4 frames de mayor confianza
        frames_with_scores.sort(key=lambda x: x["max_score"], reverse=True)
        best_frames_bytes = [f["frame_bytes"] for f in frames_with_scores[:4]]
        
        # Construir lista de DetectionResult para la tabla del PDF
        detections_list = []
        for det_data in aggregated_detections.values():
            detections_list.append(
                DetectionResult(
                    label=det_data["label"],
                    confidence=round(det_data["confidence"], 4),
                    bbox=BoundingBox(
                        x1=int(det_data["bbox"][0]),
                        y1=int(det_data["bbox"][1]),
                        x2=int(det_data["bbox"][2]),
                        y2=int(det_data["bbox"][3])
                    )
                )
            )
            
        print(f"[BACKEND] Reporte de video generado. Fotogramas clave extraídos: {len(best_frames_bytes)}")
        pdf_bytes = generate_pdf_report_video(best_frames_bytes, detections_list)
        
        return Response(
            content=pdf_bytes, 
            media_type="application/pdf", 
            headers={"Content-Disposition": "attachment; filename=reporte_biorisk.pdf"}
        )
        
    else:
        # LÓGICA ORIGINAL PARA IMÁGENES (SIN CAMBIOS)
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