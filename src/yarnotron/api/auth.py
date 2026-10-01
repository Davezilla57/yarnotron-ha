"""Bearer-token protection for the /v1 API.

Opt-in: with no YARNOTRON_API_TOKEN configured every request is allowed, which
is how Yarnotron has always behaved on a single machine.
"""

from __future__ import annotations

import hmac

from fastapi import Depends, HTTPException, Request, status

from yarnotron.core.config import Settings, get_settings


async def require_api_token(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> None:
    expected = settings.api_token
    if expected is None:
        return

    scheme, _, supplied = request.headers.get("authorization", "").partition(" ")
    # compare_digest so the check does not leak how many leading characters of
    # the token were right.
    if scheme.lower() == "bearer" and hmac.compare_digest(
        supplied.strip().encode("utf-8"), expected.encode("utf-8")
    ):
        return

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Missing or invalid API token",
        headers={"WWW-Authenticate": "Bearer"},
    )
