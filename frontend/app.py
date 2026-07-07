"""Streamlit Frontend for BNP Paribas Cardif Claims Management."""
import streamlit as st
import pandas as pd
import plotly.express as px
import requests

API_BASE = "http://localhost:8000"
st.set_page_config(page_title="BNP Paribas Cardif Claims", layout="wide", page_icon="\U0001f3e6")

st.markdown("""<style>
.stApp { background-color: #f8f9fa; }
.main-header { background: linear-gradient(135deg, #0033a0, #009fe3); color: white; padding: 1.5rem; border-radius: 10px; margin-bottom: 1.5rem; text-align: center; }
.main-header h1 { margin: 0; font-size: 1.8rem; }
.metric-card { background: white; padding: 1.2rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center; border-left: 4px solid #0033a0; }
.metric-card .value { font-size: 2rem; font-weight: bold; color: #0033a0; }
.metric-card .label { font-size: 0.85rem; color: #6c757d; }
.stButton button { background-color: #0033a0; color: white; }
</style>""", unsafe_allow_html=True)

if "page" not in st.session_state: st.session_state.page = "Dashboard"

with st.sidebar:
    st.markdown("### \U0001f3e6 BNP Paribas Cardif")
    st.markdown("**Claims Management Platform**")
    st.divider()
    for p in ["\U0001f4ca Dashboard", "\U0001f4cb Claims List", "\U0001f4c4 Upload", "\U0001f50d RAG", "⚠️ Fraud", "\U0001f4ac Chat"]:
        if st.button(p, use_container_width=True):
            st.session_state.page = p.split(" ", 1)[1]
    st.caption("AI-Powered by MCP, RAG, LangGraph, Multimodal")

page = st.session_state.page

if page == "Dashboard":
    st.markdown('<div class="main-header"><h1>\U0001f4ca Claims Dashboard</h1><p>Real-time overview of claims processing</p></div>', unsafe_allow_html=True)
    try:
        r = requests.get(f"{API_BASE}/api/dashboard/stats", timeout=5)
        if r.status_code == 200:
            data = r.json()
            c1, c2, c3, c4 = st.columns(4)
            with c1: st.markdown(f'<div class="metric-card"><div class="value">{data["total_claims"]}</div><div class="label">Total Claims</div></div>', unsafe_allow_html=True)
            with c2: st.markdown(f'<div class="metric-card"><div class="value">{data["pending_review_count"]}</div><div class="label">Pending Review</div></div>', unsafe_allow_html=True)
            with c3: st.markdown(f'<div class="metric-card"><div class="value">€{data["total_amount_claimed"]:,.0f}</div><div class="label">Total Claimed</div></div>', unsafe_allow_html=True)
            with c4: st.markdown(f'<div class="metric-card"><div class="value">{data["average_fraud_score"]:.3f}</div><div class="label">Avg Fraud Score</div></div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            if data.get("claims_by_status"):
                df1 = pd.DataFrame(list(data["claims_by_status"].items()), columns=["Status","Count"])
                col1.plotly_chart(px.pie(df1, values="Count", names="Status", title="Claims by Status", color_discrete_sequence=px.colors.sequential.Blues_r), use_container_width=True)
            if data.get("claims_by_category"):
                df2 = pd.DataFrame(list(data["claims_by_category"].items()), columns=["Category","Count"])
                col2.plotly_chart(px.bar(df2, x="Category", y="Count", title="Claims by Category", color="Count", color_continuous_scale="Blues"), use_container_width=True)
            if data.get("monthly_volume"):
                st.plotly_chart(px.line(pd.DataFrame(data["monthly_volume"]), x="month", y="count", title="Monthly Volume", markers=True), use_container_width=True)
    except Exception as e:
        st.error(f"Connection error: {e}")

elif page == "Claims List":
    st.markdown('<div class="main-header"><h1>\U0001f4cb Claims List</h1></div>', unsafe_allow_html=True)
    try:
        r = requests.get(f"{API_BASE}/api/claims?limit=50", timeout=5)
        if r.status_code == 200:
            df = pd.DataFrame(r.json())
            if not df.empty:
                st.dataframe(df[["claim_number","policyholder_name","category","status","amount_claimed","fraud_score"]], use_container_width=True, height=500)
    except Exception as e:
        st.error(f"Error: {e}")

elif page == "Upload":
    st.markdown('<div class="main-header"><h1>\U0001f4c4 Document Upload</h1></div>', unsafe_allow_html=True)
    cid = st.number_input("Claim ID", min_value=1, step=1)
    doctype = st.selectbox("Type", ["claim_form","police_report","medical_report","photo","invoice","other"])
    f = st.file_uploader("File", type=["pdf","jpg","png","csv"])
    if f and st.button("Upload"):
        r = requests.post(f"{API_BASE}/api/claims/{cid}/documents", files={"file": (f.name, f.getvalue(), f.type)}, data={"doc_type": doctype}, timeout=30)
        if r.ok:
            st.success("Uploaded!")
            if st.button("Run Analysis"):
                r2 = requests.post(f"{API_BASE}/api/claims/{cid}/documents/analyze", timeout=30)
                st.success("Analysis done!" if r2.ok else "Analysis failed")
        else:
            st.error(f"Failed: {r.text}")

elif page == "RAG":
    st.markdown('<div class="main-header"><h1>\U0001f50d RAG Semantic Search</h1></div>', unsafe_allow_html=True)
    q = st.text_input("Search query", placeholder="e.g., high-value property claim")
    col = st.selectbox("Collection", ["historical_claims","insurance_policies","few_shot_examples"])
    if q and st.button("Search"):
        r = requests.get(f"{API_BASE}/api/claims/rag-query", params={"q": q, "top_k": 5, "collection": col}, timeout=10)
        if r.ok:
            for res in r.json().get("results", []):
                with st.expander(f"{res['id']} (score: {res['score']:.4f})"):
                    st.json(res.get("metadata", {}))
                    st.write(res.get("document","")[:400])
        else:
            st.error("Query failed")

elif page == "Fraud":
    st.markdown('<div class="main-header"><h1>⚠️ Fraud Analysis</h1></div>', unsafe_allow_html=True)
    try:
        r = requests.get(f"{API_BASE}/api/claims?limit=50", timeout=5)
        if r.ok:
            df = pd.DataFrame(r.json())
            if not df.empty:
                fig = px.scatter(df, x="amount_claimed", y="fraud_score", color="fraud_score", hover_name="claim_number", title="Fraud Risk Analysis", color_continuous_scale="RdYlGn_r")
                st.plotly_chart(fig, use_container_width=True)
                high = df[df["fraud_score"] > 0.5]
                for _, c in high.iterrows():
                    st.warning(f"{c['claim_number']} | {c['policyholder_name']} | Score: {c['fraud_score']:.3f}")
    except Exception as e:
        st.error(f"Error: {e}")

elif page == "Chat":
    st.markdown('<div class="main-header"><h1>\U0001f4ac Claims Assistant</h1></div>', unsafe_allow_html=True)
    if "chat_h" not in st.session_state: st.session_state.chat_h = []
    for m in st.session_state.chat_h:
        with st.chat_message(m["role"]):
            st.write(m["content"])
    inp = st.chat_input("Ask about claims, fraud, documents...")
    if inp:
        st.session_state.chat_h.append({"role": "user", "content": inp})
        try:
            r = requests.post(f"{API_BASE}/api/chat", json={"message": inp, "session_id": "streamlit"}, timeout=10)
            resp = r.json().get("response", "") if r.ok else "Error"
            st.session_state.chat_h.append({"role": "assistant", "content": resp})
        except Exception as e:
            st.session_state.chat_h.append({"role": "assistant", "content": f"Error: {e}"})
        st.rerun()

elif page == "Upload":
    pass
