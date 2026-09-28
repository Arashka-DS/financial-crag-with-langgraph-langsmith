import os
import time
import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from langchain_core.tracers.context import collect_runs
from src.graph import crag_app

app = FastAPI(title="Financial Market Corrective RAG (CRAG) Engine", version="1.1.0")

class QueryRequest(BaseModel):
    question: str = Field(..., example="What is the maximum daily fiat on-ramp limit under Circular 402/12?")

class CitationItem(BaseModel):
    source_doc_id: int
    exact_quote: str

class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationItem]
    hallucination_passed: bool
    retries: int
    execution_trace: list[str]
    latency_ms: float
    langsmith_run_url: str | None

@app.post("/query-crag", response_model=QueryResponse)
def execute_crag(request: QueryRequest):
    t0 = time.perf_counter()
    
    initial_state = {
        "question": request.question,
        "transformed_query": "",
        "documents": [],
        "generation": "",
        "citations": [],
        "doc_relevance_passed": False,
        "hallucination_check_passed": False,
        "retry_count": 0,
        "execution_trace": []
    }

    try:
        # Collect LangSmith trace context dynamically
        with collect_runs() as cb:
            final_state = crag_app.invoke(initial_state)
            run_id = str(cb.traced_runs[0].id) if cb.traced_runs else None

        latency = (time.perf_counter() - t0) * 1000
        
        project_name = os.getenv("LANGCHAIN_PROJECT", "financial-crag-engine")
        run_url = f"https://smith.langchain.com/o/default/projects/p/{project_name}/r/{run_id}" if run_id else None

        return QueryResponse(
            answer=final_state["generation"],
            citations=final_state.get("citations", []),
            hallucination_passed=final_state["hallucination_check_passed"],
            retries=final_state["retry_count"],
            execution_trace=final_state["execution_trace"],
            latency_ms=round(latency, 2),
            langsmith_run_url=run_url
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CRAG Pipeline Error: {str(e)}")

@app.get("/health")
def health():
    return {
        "status": "OPERATIONAL",
        "vector_backend": "PostgreSQL 16 + pgvector (IVFFlat)",
        "langsmith_tracing": os.getenv("LANGCHAIN_TRACING_V2", "false")
    }
