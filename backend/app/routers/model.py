from fastapi import APIRouter
from app.services.yolo_service import yolo_service
from app.schemas.detection import ClassesResponse

router = APIRouter(prefix="/model", tags=["Model"])

@router.get("/classes", response_model=ClassesResponse)
def get_classes():
    return ClassesResponse(classes=yolo_service.get_classes())