import streamlit as st
import requests

st.set_page_config(page_title="Financial CRAG Inspector", layout="wide")

st.markdown("""
    <style>
    .node-box { background-color: #0F172A; padding: 12px; border-radius: 8px; border: 1px solid #1E293B; margin-bottom: 8px; }
    .trace-step { font-family: monospace; font-size: 12px; color: #38BDF8; }
    .quote-box { background-color: #1E293B; border-left: 3px solid #38BDF8; padding: 8px 12px; margin-top: 5px; }
    </style>
""", unsafe_allow_html=True)

st.title("🏛️ Financial Regulatory Corrective RAG (CRAG)")
st.caption("Self-Correcting State Machine with Hybrid pgvector Search, Citation Tracking & LangSmith Telemetry")

user_query = st.text_input(
    "Enter Compliance / Regulatory Query:",
    value="What is the maximum daily fiat on-ramp limit under Circular 402/12 for verified users?"
)

if st.button("Execute Corrective RAG Graph", type="primary"):
    with st.spinner("Executing State Graph (Hybrid Search -> Grade -> Rewrite -> Generate -> Audit)..."):
        try:
            res = requests.post("http://crag_api:8000/query-crag", json={"question": user_query})
            res.raise_for_status()
            data = res.json()

            col1, col2 = st.columns([1.4, 1])

            with col1:
                st.subheader("📋 Verified Financial Synthesis")
                st.write(data["answer"])

                if data.get("citations"):
                    st.markdown("##### 📌 Exact Evidence & Citations")
                    for idx, cit in enumerate(data["citations"]):
                        st.markdown(f"**[{idx+1}] Source Document ID: `{cit['source_doc_id']}`**")
                        st.markdown(f"<div class='quote-box'><em>\"{cit['exact_quote']}\"</em></div>", unsafe_allow_html=True)

                st.divider()
                m1, m2, m3 = st.columns(3)
                m1.metric("Pipeline Latency", f"{data['latency_ms']} ms")
                m2.metric("Correction Loops", data["retries"])
                m3.metric("Hallucination Audit", "PASSED" if data["hallucination_passed"] else "FAILED")

                if data.get("langsmith_run_url"):
                    st.link_button("🔗 Inspect Full Trace in LangSmith", data["langsmith_run_url"])

            with col2:
                st.subheader("🔍 LangGraph State Transitions")
                for step in data["execution_trace"]:
                    st.markdown(f"<div class='node-box'><span class='trace-step'>⚡ {step}</span></div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Execution Error: {e}")
