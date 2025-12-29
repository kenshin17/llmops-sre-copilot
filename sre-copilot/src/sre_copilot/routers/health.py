from fastapi import APIRouter

router = APIRouter()


@router.get("/ready")
async def ready() -> dict[str, str]:
    return {"status": "ok"}
