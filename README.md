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

## 🚀 Quick Start (Local LLM Setup)
This pipeline is configured to run 100% locally and privately using Ollama, avoiding paid API lock-in.

1. **Install Ollama** on your host machine.
   
2. **Pull the required local models** by running these commands in your terminal:
   ```bash
   ollama run llama3.2
   ollama pull nomic-embed-text
   ```
   
3. **Environment Configuration:** A pre-configured `proj.env` file is included in the root directory. It automatically routes LangChain requests to your host's local Ollama instance via Docker's internal networking.

*(Optional) If you wish to use LangSmith for graph tracing, edit `proj.env` and add your LangSmith API key*:
```bash
OPENAI_API_BASE=http://host.docker.internal:11434/v1
OPENAI_API_KEY=ollama_fake_key
LANGSMITH_API_KEY=your_langsmith_key_here
```
*(Reminder) To avoid Windows Docker networking issues, either define a Windows environment variable `OLLAMA_HOST` `0.0.0.0` or use any other similar methods that you see fit.*

4. **Spin up the Infrastructure:** `docker compose --env-file proj.env up -d --build`

5. **Access the Inspector:** Navigate to `http://localhost:8501`
