import streamlit as st
from backend.qa_pipeline import get_answer

# ----- APP CONFIG -----
st.set_page_config(page_title="TripMate - AI Transit Assistant", page_icon="🧳", layout="wide")

# ----- HEADER -----
st.markdown(
    """
    <h1 style='text-align: center; color: #4CAF50;'>🧳 TripMate</h1>
    <p style='text-align: center; font-size: 18px; color: grey;'>
    Your AI-powered public transportation assistant — plan routes, check schedules, and get real-time updates from your documents.
    </p>
    """,
    unsafe_allow_html=True
)

# ----- SESSION STATE FOR CHAT -----
if "messages" not in st.session_state:
    st.session_state.messages = []

# ----- SLIDER FOR TOP K -----
top_k = st.slider("🔍 Number of retrieved chunks (k)", 1, 8, 4)

# ----- DISPLAY CHAT HISTORY -----
def display_messages():
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f"""
                <div style='background-color:#A8E6A3; color:#222; padding:16px; border-radius:12px; margin:8px 0; text-align:right;'>
                    <span style='font-size:1.15em; font-weight:bold;'>You:</span> <span style='font-size:1.15em; font-weight:bold;'>{msg['content']}</span>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div style='background-color:#D1B3FF; color:#222; padding:16px; border-radius:12px; margin:8px 0; text-align:left;'>
                    <span style='font-size:1.15em; font-weight:bold;'>TripMate:</span> <span style='font-size:1.15em; font-weight:bold;'>{msg['content']}</span>
                </div>
                """,
                unsafe_allow_html=True
            )

display_messages()

# ----- USER INPUT -----
query = st.chat_input("Ask me anything about public transportation...")

# ----- WHEN USER SENDS A MESSAGE -----
if query:
    # Store user message
    st.session_state.messages.append({"role": "user", "content": query})

    with st.spinner("Retrieving and generating..."):
        res = get_answer(query, top_k=top_k)

    bot_reply = res["answer"]

    # Store bot reply
    st.session_state.messages.append({"role": "bot", "content": bot_reply})

    # Show retrieved chunks (expandable)
    with st.expander("📄 Retrieved chunks and sources"):
        for i, r in enumerate(res["retrieved"], 1):
            st.markdown(f"**{i}. Score:** {r['score']:.4f} — **Source:** `{r['meta']['source']}` chunk `{r['meta']['chunk_id']}`")
            st.write(r["text"])

    st.experimental_rerun()

# ----- FOOTER -----
st.markdown(
    "<hr><p style='text-align:center;color:grey;font-size:14px;'>© 2025 TripMate AI — Making transit smarter.</p>",
    unsafe_allow_html=True
)
