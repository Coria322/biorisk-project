from ultralytics import YOLO
import json

model_path = r'C:\Ingenieria_software\8vo Semestre\Proyecto_imgs\yolov8s.pt'
try:
    model = YOLO(model_path)
    print(json.dumps(model.names, indent=4))
except Exception as e:
    print(f"Error: {e}")
