"""API key authentication for ChainCommand."""

from __future__ import annotations

import hmac
import logging
import warnings

from fastapi import HTTPException, Request, WebSocket, WebSocketException, status

from .config import settings

_auth_log = logging.getLogger(__name__)


def require_api_key(request: Request) -> None:
    """FastAPI dependency that validates the X-API-Key header."""
    key = request.headers.get("X-API-Key", "")
    if not hmac.compare_digest(key, settings.api_key.get_secret_value()):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


async def check_ws_query_key(websocket: WebSocket) -> bool:
    """Check WebSocket query-param API key (deprecated path).

    Returns True if the client was authenticated via query param,
    False if no query param was provided (caller should use message-based auth).
    Raises WebSocketException if a query param was provided but invalid.

    Preferred flow (secure):
        1. Client connects without query params.
        2. Server accepts the connection.
        3. Client sends ``{"type": "auth", "api_key": "..."}`` as the first
           message.
        4. Server validates and proceeds, or closes with 1008.

    Deprecated fallback:
        Passing ``?api_key=`` as a query parameter still works but logs a
        deprecation warning because query strings are visible in server logs,
        proxy logs, and browser history.
    """
    # --- Deprecated fallback: query-param auth ---
    qp_key = websocket.query_params.get("api_key", "")
    if qp_key:
        warnings.warn(
            "Passing api_key as a WebSocket query parameter is deprecated and "
            "insecure (logged by proxies). Send a JSON auth message as the "
            'first frame instead: {"type": "auth", "api_key": "..."}',
            DeprecationWarning,
            stacklevel=2,
        )
        _auth_log.warning(
            "ws_auth_query_param_deprecated: client used query-string API key"
        )
        if not isinstance(qp_key, str):
            raise WebSocketException(
                code=status.WS_1008_POLICY_VIOLATION, reason="Invalid API key"
            )
        if hmac.compare_digest(qp_key, settings.api_key.get_secret_value()):
            return True  # authenticated via deprecated path
        raise WebSocketException(
            code=status.WS_1008_POLICY_VIOLATION, reason="Invalid API key"
        )

    # No query-param key provided — caller should use message-based auth
    return False


async def authenticate_ws_first_message(websocket: WebSocket) -> bool:
    """Wait for the first WebSocket message and validate it as an auth frame.

    Expected payload::

        {"type": "auth", "api_key": "<secret>"}

    Returns ``True`` on success.  On failure, closes the socket and returns
    ``False``.
    """
    import asyncio
    import json

    try:
        raw = await asyncio.wait_for(websocket.receive_text(), timeout=10.0)
    except asyncio.TimeoutError:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Auth timeout")
        return False
    except Exception:
        return False

    try:
        msg = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Invalid auth frame")
        return False

    if not isinstance(msg, dict) or msg.get("type") != "auth":
        await websocket.close(
            code=status.WS_1008_POLICY_VIOLATION,
            reason="First message must be an auth frame",
        )
        return False

    key = msg.get("api_key", "")
    if not isinstance(key, str):
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Invalid API key")
        return False
    if not hmac.compare_digest(key, settings.api_key.get_secret_value()):
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Invalid API key")
        return False

    return True
