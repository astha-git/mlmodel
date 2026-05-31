import streamlit as st
import pandas as pd
import predictcareer as pc

st.set_page_config(page_title="Career Recommender", layout="centered")

st.title("Career Recommendation System")
st.markdown("Fill the short quiz — we'll predict careers, recommend courses/certificates, show internships, and answer questions via a simple chatbot.")

with st.expander("How it works"):
    st.write("""
    - The model is trained on quiz-style numeric scores and a target `career` label.\n
    - We return a ranked list of career suggestions and curated resources.\n
    - We also pull remote job/internship listings from Remotive (public API) and provide a simple FAQ/chat interface.
    """)

# --- Quiz inputs (match your dataset columns) ---
st.header("Quick Quiz (inputs)")
col1, col2 = st.columns(2)
with col1:
    math_score = st.slider("Math score", 1, 10, 7)
    programming_score = st.slider("Programming score", 1, 10, 6)
    creativity_score = st.slider("Creativity score", 1, 10, 6)
    communication_score = st.slider("Communication score", 1, 10, 6)
    problem_solving_score = st.slider("Problem solving score", 1, 10, 7)
with col2:
    theory_score = st.slider("Theory score", 1, 10, 6)
    interest_tech = st.slider("Interest in technology (1-10)", 1, 10, 7)
    interest_arts = st.slider("Interest in arts (1-10)", 1, 10, 4)
    interest_business = st.slider("Interest in business (1-10)", 1, 10, 3)
    E_vs_I = st.slider("E vs I (extraversion 1-10)", 1, 10, 5)
    S_vs_N = st.slider("S vs N (sensing 1-10)", 1, 10, 5)
    T_vs_F = st.slider("T vs F (thinking 1-10)", 1, 10, 5)
    J_vs_P = st.slider("J vs P (judging 1-10)", 1, 10, 5)


if st.button("Predict career"):
    input_payload = {
        "math_score": math_score,
        "programming_score": programming_score,
        "creativity_score": creativity_score,
        "communication_score": communication_score,
        "problem_solving_score": problem_solving_score,
        "theory_score": theory_score,
        "interest_tech": interest_tech,
        "interest_arts": interest_arts,
        "interest_business": interest_business,
        "E_vs_I": E_vs_I,
        "S_vs_N": S_vs_N,
        "T_vs_F": T_vs_F,
        "J_vs_P": J_vs_P
    }
    try:
        preds = pc.predict_from_dict(input_payload, top_k=5)
    except Exception as e:
        st.error("Model not loaded or error: %s. Please run train.py to create the model artifact first." % e)
        preds = []

    if preds:
        st.subheader("Top career suggestions")
        for p in preds:
            st.write(f"- **{p['career']}** — {p['prob']*100:.1f}%")

        top_careers = [p["career"] for p in preds]
        st.subheader("Recommended courses & certifications")
        resources = pc.get_resources_for_careers(top_careers)
        for c in top_careers:
            st.markdown(f"**{c}**")
            items = resources.get(c, [])
            if not items:
                st.write("No curated resources available. Consider searching online.")
            else:
                for it in items:
                    st.write(f"- {it.get('title')} ({it.get('type')}) — {it.get('url')}")

        st.subheader("Remote job / internship listings (Remotive)")
        for c in top_careers[:3]:
            st.markdown(f"**Jobs for: {c}**")
            jobs = pc.get_jobs_from_remotive(c, limit=5)
            if not jobs:
                st.write("No jobs found or failed to fetch.")
            else:
                for j in jobs:
                    if "error" in j:
                        st.write(j["error"])
                    else:
                        st.write(f"- [{j['title']}]({j['url']}) — {j.get('company')} ({j.get('location')})")

# --- Chatbot ---
st.header("Career Chatbot (FAQ + semantic match)")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Ask a career question or request resources:", value="", key="user_input")
if st.button("Ask"):
    if user_input.strip():
        ans = pc.chat_answer(user_input)
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", ans))
    else:
        st.warning("Please type a question.")

for speaker, text in st.session_state.chat_history[-10:]:
    if speaker == "You":
        st.markdown(f"**You:** {text}")
    else:
        st.markdown(f"**Bot:** {text}")

st.caption("Note: This chatbot is a simple helper")
