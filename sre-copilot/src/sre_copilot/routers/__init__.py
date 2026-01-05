from fastapi import APIRouter

from sre_copilot.routers.health import router as health_router
from sre_copilot.routers.retrieval import router as retrieval_router


def create_api() -> APIRouter:
    api = APIRouter()
    api.include_router(health_router, prefix="/health", tags=["health"])
    api.include_router(retrieval_router, prefix="/v1", tags=["retrieval"])
    return api
