
import streamlit as st
import json
from openai import OpenAI, RateLimitError
from sentence_transformers import SentenceTransformer, util

# Page setup
st.set_page_config(page_title="LegalMate – AI Legal Assistant", layout="centered")
st.title("💬 LegalMate – AI Legal Assistant")
st.caption("Ask questions about VCAT and consumer rights in Victoria. Powered by hybrid AI + FAQs.")

# Load OpenAI key securely
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Load FAQ data
with open("legal_faqs.json", "r", encoding="utf-8") as f:
    faq_data = json.load(f)

# Prepare embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
questions = [item["question"] for item in faq_data]
question_embeddings = model.encode(questions, convert_to_tensor=True)

# Chat history setup
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Show previous messages
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

# Chat input
user_prompt = st.chat_input("Ask your legal question...")

if user_prompt:
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Semantic similarity search
    query_embedding = model.encode(user_prompt, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, question_embeddings)[0]
    best_match_idx = int(scores.argmax())
    confidence = float(scores[best_match_idx])

    try:
        if confidence > 0.5:
            answer = faq_data[best_match_idx]["answer"]
            source = faq_data[best_match_idx]["source"]
            bot_reply = f"{answer}<br><a href='{source}' target='_blank'>🔗 View Source</a>"
        else:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant for Australian consumer rights and VCAT law."},
                    {"role": "user", "content": user_prompt}
                ]
            )
            bot_reply = response.choices[0].message.content
            bot_reply += "<br><small><i>Note: This response was generated using GPT and may not reflect official legal guidance.</i></small>"

    except RateLimitError:
        bot_reply = "⚠️ I'm receiving too many requests at the moment. Please wait a few seconds and try again."

    st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})
    with st.chat_message("assistant"):
        st.markdown(bot_reply, unsafe_allow_html=True)

# Disclaimer
st.markdown("---")
st.markdown("<small><i>⚠️ This chatbot provides general legal information only and does not constitute legal advice.</i></small>", unsafe_allow_html=True)
