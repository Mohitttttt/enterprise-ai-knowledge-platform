from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings

from app.api.routes.health import router as health_router
from app.api.routes.documents import router as documents_router
from app.api.routes.search import router as search_router
from app.api.routes.ask import router as ask_router
from app.api.routes.conversations import router as conversations_router


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="AI-powered document search and knowledge retrieval API.",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_url],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(health_router)
app.include_router(documents_router)
app.include_router(search_router)
app.include_router(ask_router)
app.include_router(conversations_router)


@app.get("/")
async def root():
    return {
        "message": settings.app_name,
        "status": "running",
        "version": settings.app_version,
    }