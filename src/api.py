from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.graph import crag_engine

app = FastAPI(title="Financial CRAG Engine API", version="1.0.0")

class QueryPayload(BaseModel):
    question: str

@app.post("/query")
def execute_crag(payload: QueryPayload):
    try:
        # Initialize LangGraph state
        initial_state = {
            "question": payload.question, 
            "retry_count": 0, 
            "web_fallback": False
        }
        
        # Execute the self-correcting cyclic graph
        final_state = crag_engine.invoke(initial_state)
        
        return {
            "generation": final_state.get("generation"),
            "citations": final_state.get("citations", []),
            "retry_count": final_state.get("retry_count", 0),
            "documents_retrieved": len(final_state.get("documents", [])),
            "triggered_fallback": final_state.get("web_fallback", False)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ACTIVE", "engine_loaded": crag_engine is not None}
