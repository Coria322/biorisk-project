import cv2
import numpy as np
from fastapi import UploadFile, HTTPException

async def decode_image(file: UploadFile) -> np.ndarray:
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")
    
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="No se pudo procesar la imagen")

    # Redimensionar la imagen para que el lado más largo sea de 800 px (mantiene proporción)
    h, w = frame.shape[:2]
    max_dim = max(h, w)
    if max_dim != 800:
        scale = 800 / max_dim
        new_w, new_h = int(w * scale), int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)

    return frame