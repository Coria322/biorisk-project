import os
from ultralytics import YOLO

def main():
    base_dir = os.path.dirname(__file__)
    data_yaml = os.path.join(base_dir, "datasets", "data.yaml")

    # Validación
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"No se encontró data.yaml en: {data_yaml}")

    print("Usando dataset en:", data_yaml)

    # Modelo
    model = YOLO("yolov8s.pt")

    # Entrenamiento
# Entrenamiento Optimizado
    model.train(
        data=data_yaml,
        epochs=200,          # Aumentar a 200, el modelo aún está aprendiendo en la epoca 100
        imgsz=800,           # SUBIR RESOLUCIÓN: De 640 a 800. 
                            # Crucial para ver detalles morfológicos (pelos, manchas).
        batch=-1,            # Modo automático: usará el máximo que la GPU soporte
        patience=30,         # Si no mejora en 30 épocas, se detiene solo
        optimizer="AdamW",   # AdamW, es ideal para esto
        augment=True,        # Asegurar que las aumentaciones estén activas
        cls=1.5,             # Penaliza más los errores de clasificación
        box=7.5              # Mejora el ajuste de la "caja" alrededor del insecto
    )

    print("--- Validación ---")
    model.val()

    model.export(format="onnx")

if __name__ == "__main__":
    main()