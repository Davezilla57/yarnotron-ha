"""Sign-in for the /v1 API and the dashboard.

Opt-in: with no YARNOTRON_API_TOKEN configured every /v1 request is allowed, which
is how Yarnotron has always behaved on a single machine. Two credentials are accepted
when set: the bearer token (Home Assistant and other programs) and the dashboard
username and password (people, through the browser's sign-in prompt).
"""

from __future__ import annotations

import base64
import binascii
import hmac

from fastapi import Depends, HTTPException, Request, status

from yarnotron.core.config import Settings, get_settings


def _same(supplied: str, expected: str) -> bool:
    # compare_digest so the check does not leak how many leading characters were right.
    return hmac.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8"))


def _bearer_ok(header: str, settings: Settings) -> bool:
    scheme, _, supplied = header.partition(" ")
    return bool(settings.api_token) and scheme.lower() == "bearer" and _same(supplied.strip(), settings.api_token)


def _basic_ok(header: str, settings: Settings) -> bool:
    scheme, _, supplied = header.partition(" ")
    if settings.dashboard_password is None or scheme.lower() != "basic":
        return False
    try:
        user, _, password = base64.b64decode(supplied.strip(), validate=True).decode("utf-8").partition(":")
    except (binascii.Error, UnicodeDecodeError):
        return False
    # Both are always compared, so a wrong user name takes as long as a wrong password.
    user_ok = _same(user, settings.dashboard_user)
    password_ok = _same(password, settings.dashboard_password)
    return user_ok and password_ok


def _challenge(settings: Settings) -> dict[str, str]:
    # Only a password makes the browser show its sign-in box; a bearer challenge would not.
    if settings.dashboard_password is not None:
        return {"WWW-Authenticate": 'Basic realm="Yarnotron", charset="UTF-8"'}
    return {"WWW-Authenticate": "Bearer"}


async def require_api_token(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> None:
    if settings.api_token is None and settings.dashboard_password is None:
        return
    header = request.headers.get("authorization", "")
    if _bearer_ok(header, settings) or _basic_ok(header, settings):
        return
    if settings.api_token is None:
        # Only a dashboard password is set: the old open API is not closed by adding it.
        return
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Missing or invalid credentials",
        headers=_challenge(settings),
    )


async def require_dashboard_login(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> None:
    """The dashboard page itself: a username and password when one is configured."""
    if settings.dashboard_password is None:
        return
    if _basic_ok(request.headers.get("authorization", ""), settings):
        return
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Sign in to see the dashboard",
        headers=_challenge(settings),
    )
