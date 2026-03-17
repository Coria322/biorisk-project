from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import detection, model

app = FastAPI(
    title="BioRisk Vision API",
    description="Detección de fauna de importancia médica con YOLOv8",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # puerto de Vite
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(detection.router)
app.include_router(model.router)

@app.get("/")
def root():
    return {"status": "ok", "project": "BioRisk Vision API"} 