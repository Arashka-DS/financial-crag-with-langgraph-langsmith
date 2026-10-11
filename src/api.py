import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.graph import crag_engine

app = FastAPI(title="Financial CRAG Engine API", version="1.0.0")

class QueryPayload(BaseModel):
    question: str

@app.post("/query")
def execute_crag(payload: QueryPayload):
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return {
            "generation": "SYSTEM OFFLINE: API key is missing. Configure your .env file.",
            "citations": [],
            "retry_count": 0,
            "documents_retrieved": 0,
            "triggered_fallback": False
        }
        
    initial_state = {
        "question": payload.question, 
        "retry_count": 0, 
        "web_fallback": False
    }
    
    # Let FastAPI natively handle any crashes here so it prints to the Docker log
    result = crag_engine.invoke(initial_state)
    
    return {
        "generation": result.get("generation", "No answer generated."),
        "citations": result.get("citations", []),
        "retry_count": result.get("retry_count", 0),
        "documents_retrieved": len(result.get("documents", [])),
        "triggered_fallback": result.get("web_fallback", False)
    }

@app.get("/health")
def health_check():
    return {"status": "ACTIVE", "engine_loaded": crag_engine is not None}
