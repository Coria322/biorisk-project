import os
import torch
from ultralytics import YOLO

def main():
    base_dir = os.path.dirname(__file__)
    data_yaml = os.path.abspath(os.path.join(base_dir, "./dataset_download/data.yaml"))

    # Validación
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"No se encontró data.yaml en: {data_yaml}")

    print("Usando dataset en:", data_yaml)
    
    # Validación de CUDA
    if not torch.cuda.is_available():
        print("--- ERROR: CUDA NO ESTÁ DISPONIBLE ---")
        print("El entrenamiento en CPU será extremadamente lento. Abortando.")
        return

    print(f"CUDA Detectado: {torch.cuda.get_device_name(0)}")

    # Modelo
    model = YOLO("yolov8s.pt")

    # Entrenamiento
# Entrenamiento Optimizado
    model.train(
        data=data_yaml,
        epochs=300,          # el modelo aún está aprendiendo en la epoca 100
        imgsz=800,           # SUBIR RESOLUCIÓN a 800. 
        batch=-1,            # Modo automático: usará el máximo que la GPU soporte
        patience=30,         # Si no mejora en 30 épocas, se detiene solo
        optimizer="AdamW",   # AdamW, es ideal para esto
        name="Biorisk_Detector_v2_full_ds_normalizado",
        workers=8,
        augment=True,        # Asegurar que las aumentaciones estén activas
        cls=1.5,             # Penaliza más los errores de clasificación
        box=7.5              # Mejora el ajuste de la "caja" alrededor del insecto
    )

    print("--- Validación ---")
    model.val()

    model.export(format="onnx")

if __name__ == "__main__":
    main()