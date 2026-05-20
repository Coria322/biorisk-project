from ultralytics import YOLO

# 1. Carga tu modelo entrenado
model_path = "./best_v2.pt"  # Asegúrate de que la ruta sea correcta
model = YOLO(model_path)

# 2. Extraer el diccionario de clases
clases_del_modelo = model.names

print("=============================================")
print(f" EL MODELO ES CAPAZ DE DETECTAR {len(clases_del_modelo)} CLASES:")
print("=============================================")
for id_clase, nombre_clase in clases_del_modelo.items():
    print(f" -> ID {id_clase}: {nombre_clase}")
print("=============================================")