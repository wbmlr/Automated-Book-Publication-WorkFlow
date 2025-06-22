from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
import uuid

from graph_workflow import app
from storage.database import SessionLocal, init_db, ScrapedContent
from scraper.content_fetcher import fetch_content_and_screenshot
from storage.chromadb_manager import store_final_version, query_collection
from retrieval.rl_agent import ContextualBanditAgent # MODIFIED: Import agent

# --- Initializations ---
init_db()
api = FastAPI()
api.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
rl_agent = ContextualBanditAgent(actions=["summary", "characters", "style", "setting", "plot"]) # NEW: Instantiate RL agent

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

class RetrieveRequest(BaseModel): query: str; n_results: int

class RateRequest(BaseModel): query: str; action: str; rating: int # NEW: Model for rating

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
    store_final_version("approved_versions", doc_id, req.content)
    return {"status": "approved", "doc_id": doc_id}

@api.get("/api/view-postgres")
def view_postgres_cache(db: Session = Depends(get_db), limit: int = Query(5, ge=1, le=100)):
    items = db.query(ScrapedContent).order_by(ScrapedContent.id.desc()).limit(limit).all()
    return [{"id": item.id, "url": item.url} for item in items]

@api.post("/api/retrieve-chroma")
def retrieve_from_chroma(req: RetrieveRequest):
    action_keyword = rl_agent.choose_action(req.query)
    enhanced_query = f"{req.query} {action_keyword}".strip()
    print(f"RL Agent Used: '{action_keyword}' -> Query: '{enhanced_query}'")
    results = query_collection("approved_versions", enhanced_query, n_results=req.n_results)
    return {
        "results": results,
        "action_keyword": action_keyword,
        "query": req.query,
        "enhanced_query": enhanced_query # ADD THIS LINE
    }

@api.post("/api/rate-retrieval") # NEW: Endpoint for RL feedback
def rate_retrieval(req: RateRequest):
    reward = (req.rating - 2.5) / 2.5  # Normalize 0-5 rating to -1.0 to 1.0 reward
    rl_agent.update(req.query, req.action, reward)
    rl_agent.save_policy()
    return {"status": "updated", "reward": reward}

# Add at the end of api.py

@api.get("/api/get-policy")
def get_policy():
    """Extracts a human-readable version of the RL policy."""
    try:
        policy = {}
        # Ensure vectorizer is fitted by checking for vocabulary
        if not hasattr(rl_agent.vectorizer, 'vocabulary_'):
            return {"error": "Policy not trained yet."}
            
        feature_names = rl_agent.vectorizer.get_feature_names_out()
        
        for action, model in rl_agent.models.items():
            # Ensure model is fitted by checking for coefficients
            if hasattr(model, 'coef_'):
                # Sort weights to show most influential words first
                weights = sorted(zip(model.coef_, feature_names), reverse=True)
                policy[action] = {word: round(coef, 4) for coef, word in weights if abs(coef) > 0.01}
            else:
                policy[action] = "Not trained yet"
                
        return policy
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
# ----------------------------------------------------