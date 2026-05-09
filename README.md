# biorisk project

Proyecto de reconocimiento de imágenes con:

- Frontend: React + TypeScript + Vite
- Backend: FastAPI
- Modelo: (Pendiente integración), tentativamente YOLO

## Instalación

### Frontend
cd biorisk
npm install
npm run dev

### Backend
cd .. (posicionarse en la raiz del proyecto)
python -m venv .venv
source .venv/bin/activate en linux  # o .venv\Scripts\activate en Windows

cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload

debe mostrar un mensaje en json. al consultar el puerto de vite debe mostrar la app en linea y permitir la interacción