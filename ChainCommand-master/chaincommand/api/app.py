"""FastAPI application for ChainCommand dashboard and control."""

from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .. import __version__
from ..auth import authenticate_ws_first_message, check_ws_query_key
from ..config import settings
from ..utils.logging_config import get_logger

log = get_logger(__name__)

# ── Rate limiting ────────────────────────────────────────────
# NOTE: This in-memory rate limiter assumes a single-worker process.
# For multi-worker deployments, use a shared store (e.g., Redis).

_rate_limit_store: dict[str, list[float]] = defaultdict(list)
_rate_limit_call_count: int = 0


def _check_rate_limit(request: Request) -> None:
    """Simple in-memory per-IP rate limiter with periodic stale-IP cleanup."""
    global _rate_limit_call_count

    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    window = _rate_limit_store[client_ip]
    _rate_limit_store[client_ip] = [t for t in window if t > now - 60]

    # Remove the key entirely if the window is now empty (TTL cleanup)
    if not _rate_limit_store[client_ip]:
        del _rate_limit_store[client_ip]

    # Periodic full sweep: every 100 calls, evict stale IPs
    _rate_limit_call_count += 1
    if _rate_limit_call_count >= 100:
        _rate_limit_call_count = 0
        stale_ips = [
            ip for ip, timestamps in _rate_limit_store.items()
            if not timestamps or timestamps[-1] < now - 60
        ]
        for ip in stale_ips:
            del _rate_limit_store[ip]

    current_window = _rate_limit_store[client_ip]
    if len(current_window) >= settings.rate_limit_per_minute:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    _rate_limit_store[client_ip].append(now)


def configure_middlewares(app: FastAPI) -> None:
    """Configure shared middleware for app instances."""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        allow_headers=["X-API-Key", "Content-Type"],
    )

    @app.middleware("http")
    async def rate_limit_middleware(request: Request, call_next):
        try:
            _check_rate_limit(request)
        except HTTPException as exc:
            return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
        return await call_next(request)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup / shutdown lifecycle."""
    from ..orchestrator import get_orchestrator

    orchestrator = get_orchestrator()
    await orchestrator.initialize()
    log.info("app_startup_complete")
    yield
    await orchestrator.shutdown()
    log.info("app_shutdown_complete")


app = FastAPI(
    title="ChainCommand",
    description="Supply Chain Risk & Inventory Ops — CP-SAT + RL + BOM + CTB",
    version=__version__,
    lifespan=lifespan,
)

# CORS — configurable origins
configure_middlewares(app)

# ── Register routers ────────────────────────────────────────
from .routes.control import router as control_router  # noqa: E402
from .routes.dashboard import router as dashboard_router  # noqa: E402

app.include_router(dashboard_router, prefix="/api")
app.include_router(control_router, prefix="/api")


@app.get("/")
async def root():
    return {
        "name": "ChainCommand",
        "version": __version__,
        "status": "running",
        "docs": "/docs",
    }


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "name": "ChainCommand", "version": __version__}


def _json_serial(obj):
    """JSON serializer for datetime and numpy objects."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    # Handle numpy types that are not natively JSON-serializable
    try:
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
    except ImportError:
        pass
    raise TypeError(f"Type {type(obj)} not serializable")


@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    """Live event stream via WebSocket.

    Authentication (in priority order):
    1. (Deprecated) ``?api_key=`` query param — validated before accept.
    2. (Preferred) First message: ``{"type": "auth", "api_key": "..."}``.

    Limitation: This is a server-push-only stream. The loop does not read
    messages from the client after authentication. If bidirectional
    communication is needed, add a ``websocket.receive_text()`` branch
    or implement ping/pong heartbeats.
    """
    # If query-param key is present, validate via deprecated path
    qp_key = websocket.query_params.get("api_key", "")
    if qp_key:
        await check_ws_query_key(websocket)
        await websocket.accept()
    else:
        # Message-based auth: accept first, then wait for auth frame
        await websocket.accept()
        if not await authenticate_ws_first_message(websocket):
            return
    log.info("ws_client_connected")

    from ..orchestrator import _runtime

    _SEEN_MAX = 5000
    try:
        # Use a deque to preserve insertion order; when trimming we
        # naturally drop the oldest entries instead of random ones.
        seen_ids_deque: deque[str] = deque(maxlen=_SEEN_MAX)
        seen_ids_set: set[str] = set()
        while True:
            await asyncio.sleep(1)

            if not _runtime.event_bus:
                continue

            events = _runtime.event_bus.recent_events[-20:]
            for evt in events:
                eid = evt.event_id
                if eid and eid not in seen_ids_set:
                    # If deque is at capacity, the oldest entry is auto-evicted
                    if len(seen_ids_deque) == _SEEN_MAX:
                        evicted = seen_ids_deque[0]  # will be popped by append
                        seen_ids_set.discard(evicted)
                    seen_ids_deque.append(eid)
                    seen_ids_set.add(eid)
                    data = evt.model_dump()
                    text = json.dumps(data, default=_json_serial)
                    await websocket.send_text(text)
    except WebSocketDisconnect:
        log.info("ws_client_disconnected")
    except Exception as exc:
        log.error("ws_error", error=str(exc))
