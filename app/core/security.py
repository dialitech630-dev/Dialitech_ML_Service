import hmac

from fastapi import Header, HTTPException


async def verify_api_key(x_api_key: str = Header(...)) -> str:
    from app.core.config import settings

    if not hmac.compare_digest(x_api_key, settings.API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return x_api_key
