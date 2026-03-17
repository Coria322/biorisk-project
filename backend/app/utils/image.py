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

    return frame