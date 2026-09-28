from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.graph import crag_engine

app = FastAPI(title="Financial CRAG Engine API", version="1.0.0")

class QueryPayload(BaseModel):
    question: str

@app.post("/query")
def execute_crag(payload: QueryPayload):
    # Defensive LLM Guardrail
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or not api_key.startswith("sk-"):
        return {
            "generation": "SYSTEM OFFLINE: OPENAI_API_KEY is missing or invalid. Please configure your .env file to enable the LangGraph LLM engine.",
            "citations": [],
            "retry_count": 0,
            "documents_retrieved": 0,
            "triggered_fallback": False
        }
        
    try:
        # Initialize LangGraph state
        initial_state = {
            "question": payload.question, 
            "retry_count": 0, 
            "web_fallback": False
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ACTIVE", "engine_loaded": crag_engine is not None}
