# Financial Market Corrective RAG (CRAG) Engine

A production-grade Generative AI platform for querying volatile financial regulations, central bank circulars, and crypto compliance documentation. This pipeline eliminates the "Dump and Pray" RAG failure mode by introducing autonomous grading, self-correction, and deterministic citation provenance.

## 🏛️ Architectural Upgrades over Naive RAG
1. **Reciprocal Rank Fusion (RRF):** Pure vector databases miss exact keyword matches (e.g., "Circular 402/B"). We utilize `pgvector` to run a single database transaction that merges dense semantic cosine similarity with sparse lexical `tsvector` BM25 ranking.
2. **LangGraph State Machine Orchestration:** 
   - **Grade Node:** Discards retrieved documents that do not actively answer the prompt.
   - **Transform Node:** Rewrites the user query and re-retrieves if the initial context was poor.
   - **Hallucination Gate:** Audits the final generation line-by-line against the source chunks before serving it to the user.
3. **Structured Citation Provenance:** The generation node uses Pydantic structured outputs to bind exact verbatim quotes and `DOC_ID`s to every claim, providing complete transparency for compliance officers.
4. **LangSmith Telemetry:** Full tracing of node transitions, token spend, and latency.

## 🚀 Quick Start
1. Configure `.env` with `OPENAI_API_KEY` and `LANGCHAIN_API_KEY`.

Environment Variables (`.env`)
```bash
OPENAI_API_KEY=sk-...
LANGCHAIN_API_KEY=lsv2_...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=FinTech_CRAG
```

2. Spin up the Postgres vector infrastructure: `docker-compose up -d --build`
3. Access the Streamlit Execution Inspector at `http://localhost:8501`.
4. Enter a regulatory query. Monitor the LangGraph trace on the right panel, and expand the "Source Citations" accordion to verify the exact text quotes extracted by the LLM.
