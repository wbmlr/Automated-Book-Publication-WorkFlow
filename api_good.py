from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
import uuid

from graph_workflow import app
from storage.database import SessionLocal, init_db, ScrapedContent
from scraper.content_fetcher import fetch_content_and_screenshot
# Import the function for storing documents
from storage.chromadb_manager import store_final_version

# --- Initializations ---
init_db()
api = FastAPI()
api.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()

# --- Pydantic Models ---
class StartRequest(BaseModel): url: str

class ContinueRequest(BaseModel): 
    thread_id: str
    feedback: str = ""
    llm_provider: str
    scraped_text: str
    generated_text: str

# NEW: Model for the approve endpoint
class ApproveRequest(BaseModel):
    content: str
    collection_name: str = "approved_versions"

# --- Stream Generator ---
async def stream_llm_outputs(req: ContinueRequest):
    """A generator that streams LLM outputs and prints all events for debugging."""
    config = {"configurable": {"thread_id": req.thread_id}}
    provider = req.llm_provider
    
    start_tag = f"[MODEL_START:{provider}]\n"
    print(f"BACKEND SENDING: {start_tag.strip()}")
    yield start_tag

    try:
        state_update = {
            "feedback": [req.feedback],
            "llm_provider": provider,
            "scraped_text": req.scraped_text,
            "generated_text": req.generated_text,
        }
        
        async for event in app.astream_events(state_update, config=config, version="v1"):
            # --- THIS IS THE DEBUGGING PRINT STATEMENT ---
            # print("EVENT RECEIVED:", event)
            # ---------------------------------------------
            
            # The original filter (which is likely wrong) remains for now.
            if event["event"] == "on_chain_stream" and event["name"] == "generator":
                content_chunk = event["data"]["chunk"].get("spun_content")
                if content_chunk:
                    print(f"BACKEND SENDING: '{content_chunk}'")
                    yield content_chunk
    except Exception as e:
        print(f"ERROR streaming from {provider}: {e}")
        yield f"\n[ERROR] Failed from {provider}.\n"
        
    end_tag = f"\n[MODEL_END]\n"
    print(f"BACKEND SENDING: {end_tag.strip()}")
    yield end_tag

# --- API Endpoints ---
@api.post("/api/start")
def start_workflow(req: StartRequest, db: Session = Depends(get_db)):
    cached = db.query(ScrapedContent).filter(ScrapedContent.url == req.url).first()
    if cached: raw_text = cached.raw_text
    else:
        scraped_data = fetch_content_and_screenshot(req.url)
        if not scraped_data: raise HTTPException(status_code=500, detail="Scrape failed.")
        new_content = ScrapedContent(url=req.url, raw_text=scraped_data["text"], screenshot=scraped_data["screenshot_bytes"])
        db.add(new_content); db.commit(); raw_text = new_content.raw_text
    return {"thread_id": str(uuid.uuid4()), "raw_content": raw_text}

@api.post("/api/continue")
async def continue_workflow(req: ContinueRequest):
    return StreamingResponse(stream_llm_outputs(req), media_type="text/event-stream")

# NEW: Endpoint for approving and saving the final version
@api.post("/api/approve")
def approve_version(req: ApproveRequest):
    doc_id = f"approved_{uuid.uuid4()}"
    store_final_version(
        collection_name=req.collection_name,
        doc_id=doc_id,
        document=req.content
    )
    return {"status": "approved", "doc_id": doc_id}


    # In api.py

import time
import asyncio

# --- ADD THIS ENTIRE BLOCK TO THE END OF THE FILE ---

async def test_stream_generator():
    """A simple async generator to test streaming."""
    for i in range(5):
        # Yield a chunk of data
        yield f"Chunk {i+1} received at {time.time()}\n"
        # Wait for 1 second to simulate a slow process
        await asyncio.sleep(1)

@api.get("/api/test-stream")
async def test_streaming_endpoint():
    """
    This endpoint is for testing purposes only.
    It streams a new chunk of data every second for 5 seconds.
    """
    return StreamingResponse(test_stream_generator(), media_type="text/plain")
# ----------------------------------------------------