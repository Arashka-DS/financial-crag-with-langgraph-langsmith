import os
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.graph import crag_app

app = FastAPI(title="Financial Market Corrective RAG (CRAG) Engine", version="1.0.0")

class QueryRequest(BaseModel):
    question: str = Field(..., example="What are the daily withdrawal limits for crypto exchanges under Central Bank Circular 402?")

class QueryResponse(BaseModel):
    answer: str
    hallucination_passed: bool
    retries: int
    execution_trace: list
    latency_ms: float
    citation: dict

@app.post("/query-crag", response_model=QueryResponse)
def execute_crag(request: QueryRequest):
    t0 = time.perf_counter()
    
    initial_state = {
        "question": request.question,
        "transformed_query": "",
        "documents": [],
        "generation": "",
        "doc_relevance_passed": False,
        "hallucination_check_passed": False,
        "answer_relevance_passed": False,
        "retry_count": 0,
        "execution_trace": []
    }

    try:
        # LangSmith automatically intercepts and streams full traces because of LANGCHAIN_TRACING_V2=true
        final_state = crag_app.invoke(initial_state)
        latency = (time.perf_counter() - t0) * 1000

        return QueryResponse(
            with col1:
                st.subheader("📋 Grounded Financial Synthesis")
                st.write(data["answer"])
                
                if data.get("citations"):
                    with st.expander("🔍 View Source Citations & Provenance"):
                        for idx, cit in enumerate(data["citations"]):
                            st.markdown(f"**[{idx+1}] Doc ID {cit['source_doc_id']}:**")
                            st.info(f"*{cit['exact_quote']}*")

                m1, m2, m3 = st.columns(3)
                m1.metric("Pipeline Latency", f"{data['latency_ms']} ms")
                m2.metric("Correction Loops", data["retries"])
                m3.metric("Hallucination Audit", "PASSED" if data["hallucination_passed"] else "FLAGGED")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CRAG Pipeline Error: {str(e)}")

@app.get("/health")
def health():
    return {
        "status": "OPERATIONAL",
        "vector_backend": "pgvector (IVFFlat)",
        "langsmith_tracing": os.getenv("LANGCHAIN_TRACING_V2", "false")
    }
