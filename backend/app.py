from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, status, Form, Request
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from typing import Optional
from contextlib import asynccontextmanager
import uvicorn
import asyncio
import threading
import queue as q_module
import os
import sys

# Add root directory to path for absolute imports
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import uuid
import secrets
import shutil
import json
from loguru import logger

from backend.main import Farm360Agent
from backend.config import settings
from backend.provider_manager import provider_manager

# ── Rate limiting ─────────────────────────────────────────────────────────────
from collections import defaultdict
import time

class RateLimiter:
    """Simple in-memory sliding-window rate limiter."""
    def __init__(self, max_requests: int = 20, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._clients: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def is_allowed(self, client_id: str) -> bool:
        now = time.time()
        cutoff = now - self.window_seconds
        with self._lock:
            timestamps = self._clients[client_id]
            # Prune old entries
            self._clients[client_id] = [t for t in timestamps if t > cutoff]
            if len(self._clients[client_id]) >= self.max_requests:
                return False
            self._clients[client_id].append(now)
            return True

rate_limiter = RateLimiter(max_requests=30, window_seconds=60)


# ── Global state ─────────────────────────────────────────────────────────────
agent: Optional[Farm360Agent] = None
TEMP_DIR = os.path.join(os.path.dirname(__file__), "temp_uploads")
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


# ── Startup / shutdown ────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("FastAPI starting — initialising Farm360 Agent…")
    os.makedirs(TEMP_DIR, exist_ok=True)
    global agent
    try:
        # Pass model_base_path from settings
        agent = Farm360Agent(
            use_mock_llm=False,
            model_base_path=settings.model_base_path,
        )
        agent.memory.set_user_profile(agent.user_id, {
            "location": "Assam, India",
            "farm_size": 100,
            "primary_crop": "Rice",
            "secondary_crops": ["Mustard", "Jute"],
            "livestock": "Mixed dairy herd (12 cattle)",
        })
        logger.success("Farm360 Agent ready — serving requests.")
    except Exception as e:
        logger.error(f"Agent boot failed: {e}")
        raise e
    yield
    logger.info("Farm360 Backend shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Farm360 AI API",
    version="3.0",
    description="Intelligent agricultural advisory — powered by OpenRouter LLMs",
    lifespan=lifespan,
)

# More restrictive CORS for production
_CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)


def verify_api_key(api_key: str = Depends(api_key_header)):
    """Verify the API key (constant-time comparison to avoid timing attacks)."""
    if not secrets.compare_digest(api_key, settings.farm360_api_key):
        logger.warning("Unauthorized access attempt.")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    return api_key


# ── Middleware: Rate Limiting ─────────────────────────────────────────────────
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    # Extract client IP
    forwarded = request.headers.get("X-Forwarded-For")
    client_ip = forwarded.split(",")[0].strip() if forwarded else request.client.host if request.client else "unknown"
    
    if not rate_limiter.is_allowed(client_ip):
        logger.warning(f"Rate limit exceeded for {client_ip}")
        return StreamingResponse(
            content=iter(['{"detail":"Rate limit exceeded. Try again later."}']),
            status_code=429,
            media_type="application/json",
            headers={"Retry-After": "60"},
        )
    
    response = await call_next(request)
    return response


# ── Input validation helpers ──────────────────────────────────────────────────
MAX_QUERY_LENGTH = 10000

def validate_query(query: str):
    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    if len(query) > MAX_QUERY_LENGTH:
        raise HTTPException(status_code=400, detail=f"Query too long (max {MAX_QUERY_LENGTH} chars)")


def sanitize_filename(original: str) -> str:
    """Generate a safe UUID-based filename to prevent path traversal."""
    ext = os.path.splitext(original)[1] if "." in original else ""
    safe_name = f"{uuid.uuid4().hex}{ext}"
    return safe_name


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/")
def health_check():
    """Health check with model availability and key pool status."""
    if agent is None:
        return {"status": "error", "message": "Agent not initialized"}

    model_status = {}
    if hasattr(agent, 'api') and agent.api is not None:
        model_status["crop_model"]   = agent.api.crop_model is not None
        model_status["dairy_model"]  = agent.api.dairy_model is not None
        model_status["vision_model"] = agent.api.vision_model is not None
        model_status["animal_model"] = agent.api.animal_model is not None

    return {
        "status": "ok",
        "message": "Farm360 AI v3 — ready",
        "llm_active": agent.has_llm if agent else False,
        "models_loaded": model_status,
        "key_pools": {
            provider: len(pool)
            for provider, pool in provider_manager._pools.items()
        },
    }


@app.get("/keys/status")
def keys_status(api_key: str = Depends(verify_api_key)):
    """
    Live view of API key pool health.
    Shows availability and cooldown state per key (redacts the actual key values).
    """
    return {
        "status": "ok",
        "pools": provider_manager.status(),
        "has_any_key": provider_manager.has_any_key,
    }


@app.get("/api/health/providers")
def health_providers(api_key: str = Depends(verify_api_key)):
    """
    Health check for API providers.
    Returns the active provider, model, and healthy status.
    """
    return provider_manager.health_status()


# ── SSE streaming chat ────────────────────────────────────────────────────────
@app.post("/chat_stream")
async def chat_stream_endpoint(
    query: str = Form(...),
    session_id: str = Form("default_session"),
    model: str = Form("google/gemma-4-26b-a4b-it:free"),
    api_key: str = Depends(verify_api_key),
):
    """
    Real Server-Sent Events endpoint.
    Streams tokens from the LLM one-by-one so the frontend can render
    them progressively (like ChatGPT / Gemini).
    """
    validate_query(query)

    logger.info(f"[STREAM] model={model} query={query[:60]!r}")

    async def event_stream():
        if agent is None or not agent.has_llm:
            # ── Fallback: stream the deterministic response character by character ──
            if agent is None:
                fallback_text = "Service is warming up. Please retry in a few seconds."
            else:
                fallback_text = "⚠️ **LLM is not configured.** Please add a valid API key."

            chunk_size = 20
            for i in range(0, len(fallback_text), chunk_size):
                chunk = fallback_text[i:i+chunk_size]
                safe = chunk.replace("\n", "\\n")
                yield f"data: {safe}\n\n"
                await asyncio.sleep(0.02)
            yield "data: [DONE]\n\n"
            return

        # ── Real LLM streaming via OpenRouter ──────────────────────────────────
        token_queue: q_module.Queue = q_module.Queue()

        def producer():
            """Run the synchronous generator in a background thread."""
            try:
                for token in agent.stream_query_prose(query, model=model):
                    token_queue.put(token)
            except Exception as e:
                logger.error(f"Producer error: {e}")
                token_queue.put(f"\n\n⚠️ **System Error**: {str(e)}")
            finally:
                token_queue.put(None)  # sentinel

        thread = threading.Thread(target=producer, daemon=True)
        thread.start()

        loop = asyncio.get_event_loop()
        try:
            while True:
                # Await the blocking queue.get with a timeout so we can detect hangs
                try:
                    token = await loop.run_in_executor(
                        None, lambda: token_queue.get(timeout=30)
                    )
                except q_module.Empty:
                    logger.error("Stream timeout — no token received for 30s")
                    yield "data: ⚠️ **Stream timeout**. Please try again.\\n\\n"
                    yield "data: [DONE]\\n\\n"
                    return

                if token is None:
                    break
                # Escape newlines so each SSE frame is on one line
                safe = token.replace("\n", "\\n")
                yield f"data: {safe}\n\n"
        except asyncio.CancelledError:
            logger.info("Client disconnected — stream cancelled.")

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",   # nginx: disable buffering
            "Connection": "keep-alive",
        },
    )


# ── Non-streaming chat (legacy) ───────────────────────────────────────────────
@app.post("/chat")
async def chat_endpoint(
    query: str = Form(...),
    model: str = Form("google/gemma-4-26b-a4b-it:free"),
    api_key: str = Depends(verify_api_key),
):
    validate_query(query)
    try:
        logger.info(f"[CHAT] query={query[:60]!r}")
        if agent is None:
            raise HTTPException(status_code=503, detail="Service is warming up. Please retry in a few seconds.")
        text = await run_in_threadpool(agent.chat_blocking, query, None, model)
        return {"query": query, "response": text}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Chat endpoint error")
        raise HTTPException(status_code=500, detail="Inference error")


# ── Image analysis ────────────────────────────────────────────────────────────
@app.post("/analyze_image")
async def analyze_image_endpoint(
    query: str = Form("Analyze this crop image and diagnose any visible diseases, deficiencies, or pest damage."),
    model: str = Form("google/gemma-4-26b-a4b-it:free"),
    image: UploadFile = File(...),
    api_key: str = Depends(verify_api_key),
):
    if not image.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    # Validate file size (read first chunk to check)
    contents = await image.read(MAX_FILE_SIZE + 1)
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail=f"File too large (max {MAX_FILE_SIZE // (1024*1024)} MB)")

    # Reset stream position after reading
    image.file.seek(0) if image.file.seekable() else None

    # Use a safe UUID-based filename to prevent path traversal
    safe_name = sanitize_filename(image.filename)
    temp_path = os.path.join(TEMP_DIR, safe_name)
    try:
        logger.info(f"[IMAGE] file={image.filename} (safe={safe_name}) query={query[:50]!r}")
        # Use async file I/O via run_in_threadpool
        await run_in_threadpool(lambda: _save_upload_sync(image.file, temp_path))

        text = await run_in_threadpool(agent.chat_blocking, query, temp_path, model)
        return {"query": query, "response": text}

    except Exception as e:
        logger.exception("Image analysis error")
        raise HTTPException(status_code=500, detail="Image processing error")
    finally:
        if os.path.exists(temp_path):
            await run_in_threadpool(os.remove, temp_path)


def _save_upload_sync(file_obj, dest_path: str):
    """Synchronous helper to write uploaded file to disk."""
    with open(dest_path, "wb") as f:
        shutil.copyfileobj(file_obj, f)


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)