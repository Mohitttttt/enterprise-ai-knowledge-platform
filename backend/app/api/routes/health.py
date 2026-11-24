from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "knowledge-platform-backend",
        "message": "Backend is running successfully."
    }