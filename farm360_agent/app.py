from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, status, Form
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from typing import Optional
from contextlib import asynccontextmanager
import uvicorn
import os
import shutil
from loguru import logger

from farm360_agent.main import Farm360Agent
from farm360_agent.config import settings

# Global Agent State
agent: Optional[Farm360Agent] = None
TEMP_DIR = os.path.join(os.path.dirname(__file__), "temp_uploads")

# Pydantic Schemas for validation Input
class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User's query for the assistant")
    session_id: str = Field("default_session", description="Unique session string")

class ChatResponse(BaseModel):
    query: str
    response: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup Engine
    logger.info("FastAPI Booting - Starting ML Lifespan Hook...")
    os.makedirs(TEMP_DIR, exist_ok=True)
    global agent
    try:
        # Fails immediately here if models or env are faulty
        agent = Farm360Agent(use_mock_llm=False)
        agent.memory.set_user_profile(agent.user_id, {"location": "Assam", "farm_size": 100, "primary_crop": "Rice"})
        logger.success("Farm360 Agent Application Complete! Serving Requests.")
    except Exception as e:
        logger.error(f"Failed to boot Farm360 Agent completely: {str(e)}")
        raise e
    yield
    # Shutdown hook
    logger.info("Shutting down Farm360 Backend Core...")

app = FastAPI(title="Farm360 Production Agent API", version="2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != settings.farm360_api_key:
        logger.warning("Unauthorized access attempt detected.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Configuration API Key",
        )
    return api_key

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Farm360 Server Production Env Active"}

@app.post("/chat")
async def chat_endpoint(query: str = Form(...), api_key: str = Depends(verify_api_key)):
    """Simple text-only chat endpoint."""
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    try:
        logger.info(f"Incoming Chat Request: {query[:50]}")
        response = await run_in_threadpool(agent.chat, query)
        return {"query": query, "response": response}
    except Exception as e:
        logger.exception("Text Chat Failure")
        raise HTTPException(status_code=500, detail="Inference Error")

@app.post("/chat_stream")
async def chat_stream_endpoint(
    query: str = Form(...),
    session_id: str = Form("default_session"),
    api_key: str = Depends(verify_api_key)
):
    from fastapi.responses import StreamingResponse
    import asyncio
    
    async def event_generator():
        try:
            logger.info(f"Incoming Streaming Request: {query[:50]}")
            response_text = await run_in_threadpool(agent.chat, query)
            # Stream line-by-line to preserve markdown structure (headers, bullets)
            lines = response_text.split("\n")
            for i, line in enumerate(lines):
                chunk = line + ("\n" if i < len(lines) - 1 else "")
                safe_chunk = chunk.replace("\n", "\\n")
                yield f"data: {safe_chunk}\n\n"
                await asyncio.sleep(0.005)
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Streaming Error: {str(e)}")
            yield f"data: [ERROR] {str(e)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/analyze_image")
async def analyze_image_endpoint(
    query: str = Form("Analyze this image."), 
    image: UploadFile = File(...),
    api_key: str = Depends(verify_api_key)
):
    """Multimodal handler for images."""
    if not image.filename:
        raise HTTPException(status_code=400, detail="Empty payload passed")
        
    temp_path = os.path.join(TEMP_DIR, image.filename)
    try:
        logger.info(f"Incoming Multimodal Request: {query[:50]}")
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        response = await run_in_threadpool(agent.chat, query, temp_path)
        return {"query": query, "response": response}
    except Exception as e:
        logger.exception("Vision Pipeline Faults")
        raise HTTPException(status_code=500, detail="Failure analyzing image structure")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
