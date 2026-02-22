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
cd ..
python -m venv .venv
source .venv/bin/activate  # o .venv\Scripts\activate en Windows
pip install -r requirements.txt
uvicorn main:app --reload

debe mostrar un mensaje en json