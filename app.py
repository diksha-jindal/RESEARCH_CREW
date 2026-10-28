"""Streamlit dashboard:  streamlit run app.py"""
import pandas as pd
import streamlit as st

from src import kb
from src.demo_pipeline import finalize, run_demo
from src.logger import LOG_PATH, ROOT

st.set_page_config(page_title="Multi-Agent Research Assistant", layout="wide")
st.title("Multi-Agent Research Assistant")
st.caption("Researcher -> Fact-Checker -> Writer -> Reviewer -> Human approval")

tab_run, tab_stats = st.tabs(["Run a research task", "Run analytics"])

with tab_run:
    c1, c2 = st.columns([3, 1])
    topic = c1.text_input("Research topic", "Agentic AI")
    fault = c2.checkbox("Inject fake claim", help="Shows the fact-checker catching an unsupported claim")
    st.caption("Knowledge base topics: " + ", ".join(d.title for d in kb.load_docs()))
    if st.button("Run agents", type="primary"):
        st.session_state.res = run_demo(topic, fault)
    res = st.session_state.get("res")
    if res:
        a, b = st.columns(2)
        a.subheader("Agent trace")
        a.dataframe(pd.DataFrame(res.steps), hide_index=True)
        b.subheader("Claims and fact-check")
        b.dataframe(pd.DataFrame([{"verified": c.verified, "support": c.support, "claim": c.text} for c in res.claims]),
                    hide_index=True)
        st.metric("Quality score", res.quality)
        for f in res.flags:
            st.warning(f)
        st.subheader("Draft brief")
        st.markdown(res.report)
        if res.status == "PENDING HUMAN REVIEW":
            x, y = st.columns(2)
            if x.button("Approve and publish"):
                st.success(f"Approved. Saved to {finalize(res, True)}")
            if y.button("Reject (keep as draft)"):
                st.info(f"Saved draft to {finalize(res, False)}")
        elif res.path:
            st.info(f"{res.status}: {res.path}")

with tab_stats:
    frames = [pd.read_csv(ROOT / "data" / "dataset_additional.csv")]
    if LOG_PATH.exists():
        frames.append(pd.read_csv(LOG_PATH))
    df = pd.concat(frames, ignore_index=True)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Agent runs logged", len(df))
    k2.metric("Avg tokens", int(df.tokens_used.mean()))
    k3.metric("Avg quality", round(df.quality_score.mean(), 2))
    k4.metric("Error rate", f"{(df.errors != 'none').mean():.1%}")
    by = df.groupby("agent_role").agg(tokens=("tokens_used", "mean"), seconds=("duration_s", "mean"),
                                      quality=("quality_score", "mean")).round(2)
    st.dataframe(by)
    st.bar_chart(by["quality"])
    st.subheader("Errors by type")
    st.bar_chart(df[df.errors != "none"].errors.value_counts())
