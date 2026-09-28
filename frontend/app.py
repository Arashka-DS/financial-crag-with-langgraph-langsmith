import streamlit as st
import requests

st.set_page_config(page_title="Financial CRAG Inspector", layout="wide")

st.markdown("""
    <style>
    .node-box { background-color: #0F172A; padding: 15px; border-radius: 8px; border: 1px solid #1E293B; margin-bottom: 10px; }
    .trace-step { font-family: monospace; font-size: 13px; color: #38BDF8; }
    </style>
""", unsafe_allow_html=True)

st.title("🏛️ Financial Regulatory Corrective RAG (CRAG)")
st.markdown("Query regulatory directives, capital limits, and crypto compliance with real-time hallucination audits.")

user_query = st.text_input(
    "Enter Compliance / Regulatory Query:",
    value="What are the maximum daily fiat transaction limits for verified accounts?"
)

if st.button("Execute Corrective RAG Graph", type="primary"):
    with st.spinner("Executing State Machine (Retrieve -> Grade -> Transform -> Generate -> Audit)..."):
        try:
            res = requests.post("http://crag_api:8000/query-crag", json={"question": user_query})
            res.raise_for_status()
            data = res.json()

            col1, col2 = st.columns([1.3, 1])

            with col1:
                st.subheader("📋 Grounded Financial Synthesis")
                st.write(data["answer"])

                m1, m2, m3 = st.columns(3)
                m1.metric("Pipeline Latency", f"{data['latency_ms']} ms")
                m2.metric("Correction Loops", data["retries"])
                m3.metric("Hallucination Audit", "PASSED" if data["hallucination_passed"] else "FLAGGED")

            with col2:
                st.subheader("🔍 LangGraph Execution Trace")
                for step in data["execution_trace"]:
                    st.markdown(f"<div class='node-box'><span class='trace-step'>⚡ {step}</span></div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error executing pipeline: {e}")
