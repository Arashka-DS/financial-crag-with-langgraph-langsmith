import streamlit as st
import requests

st.set_page_config(page_title="Financial Market CRAG Inspector", layout="wide")

API_URL = "http://crag_api:8000"

st.title("⚖️ Central Bank Regulatory CRAG Engine")
st.caption("Self-Correcting Retrieval-Augmented Generation with Deterministic Citation Provenance")

query = st.text_input("Enter regulatory compliance query:", 
                      value="What are the daily card-to-card transfer limits and crypto settlement restrictions?")

if st.button("Audit Regulations", type="primary") and query:
    with st.spinner("Executing RRF search, grading context, and auditing citations..."):
        try:
            res = requests.post(f"{API_URL}/query", json={"question": query}).json()
            
            c1, c2 = st.columns([3, 2])
            
            with c1:
                st.subheader("Audited Regulatory Finding")
                st.write(res.get("generation", "No response generated."))
                
                st.subheader("Deterministic Citation Provenance")
                citations = res.get("citations", [])
                if citations:
                    for idx, cite in enumerate(citations, 1):
                        with st.expander(f"Citation #{idx} — [{cite['source_id']}]"):
                            st.markdown(f"**Verbatim Clause:** *\"{cite['verbatim_quote']}\"*")
                            st.caption(f"**Rationale:** {cite['rationale']}")
                else:
                    st.info("No explicit citations grounded in source text.")

            with c2:
                st.subheader("Graph Execution Telemetry")
                st.metric("Query Retries", res.get("retry_count", 0), delta="Max: 2")
                st.json({
                    "retrieved_chunks": len(res.get("documents", [])),
                    "execution_status": "COMPLETED",
                    "model": "gpt-4o-mini"
                })
        except Exception as e:
            st.error(f"Failed to connect to API: {e}")
