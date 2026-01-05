from fastapi import Depends, HTTPException, Request, status

from sre_copilot.config import Settings, get_settings


async def verify_api_key(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> None:
    """
    Simple API key guard. Expects header name defined in settings.api_key_header.
    """
    header_name = settings.api_key_header
    api_key = request.headers.get(header_name)
    if settings.api_keys:
        if api_key is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Missing {header_name} header",
            )
        if api_key not in settings.api_keys:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Invalid API key",
            )

    return None
