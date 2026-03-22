from fastapi import FastAPI, File, UploadFile, Form, Depends, HTTPException, status
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
import uvicorn
import os
import shutil
from farm360_agent.main import Farm360Agent

app = FastAPI(title="Farm360 AI Agent API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.environ.get("FARM360_API_KEY", "default-secret-key")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

def get_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
        )
    return api_key

# Initialize agent globally
agent = Farm360Agent(use_mock_llm=not "GEMINI_API_KEY" in os.environ)
agent.memory.set_user_profile(agent.user_id, {"location": "Assam", "farm_size": 100, "primary_crop": "Rice"})

TEMP_DIR = os.path.join(os.path.dirname(__file__), "temp_uploads")
os.makedirs(TEMP_DIR, exist_ok=True)

@app.get("/")
def root():
    return {"message": "Welcome to Farm360 AI Agent API. Use /chat or /analyze_image"}

@app.post("/chat")
async def chat_endpoint(query: str = Form(...), api_key: str = Depends(get_api_key)):
    """Simple text-only chat endpoint."""
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    try:
        response = await run_in_threadpool(agent.chat, query)
        return {"query": query, "response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.post("/chat_stream")
async def chat_stream_endpoint(
    query: str = Form(...),
    session_id: str = Form("default_session"),
    api_key: str = Depends(get_api_key)
):
    from fastapi.responses import StreamingResponse
    import asyncio
    
    async def event_generator():
        try:
            response_text = await run_in_threadpool(agent.chat, query)
            words = response_text.split(" ")
            for i, word in enumerate(words):
                chunk = word + (" " if i < len(words) - 1 else "")
                safe_chunk = chunk.replace("\n", "\\n")
                yield f"data: {safe_chunk}\n\n"
                await asyncio.sleep(0.02)
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/analyze_image")
async def analyze_image_endpoint(
    query: str = Form("Analyze this image."), 
    image: UploadFile = File(...),
    api_key: str = Depends(get_api_key)
):
    if not image.filename:
        raise HTTPException(status_code=400, detail="No image file provided.")
    temp_path = os.path.join(TEMP_DIR, image.filename)
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        response = await run_in_threadpool(agent.chat, query, temp_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process image: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    return {"query": query, "response": response}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9999)
