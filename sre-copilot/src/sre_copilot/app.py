from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI

from sre_copilot.config import get_settings
from sre_copilot.middleware.auth import verify_api_key
from sre_copilot.observability.otel import setup_otel
from sre_copilot.routers import create_api
from sre_copilot.services.milvus_schema import ensure_collection
from sre_copilot.utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    settings = get_settings()
    logger.info("Starting SRE Copilot with Milvus at %s", settings.milvus_uri)
    setup_otel(app, settings)
    try:
        await ensure_collection(settings)
        logger.info("Milvus collection ensured: %s", settings.milvus_collection)
    except Exception as exc:  # pragma: no cover - external service
        logger.warning("Milvus unavailable or misconfigured: %s", exc)
    
    yield
    
    # Shutdown (if needed in the future)


def create_app() -> FastAPI:
    app = FastAPI(
        title="AI SRE Copilot",
        description="Gateway, guardrails, routing, and retrieval demo.",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.include_router(create_api())

    return app


app = create_app()
