import streamlit as st
import pickle
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

# Page config
st.set_page_config(
    page_title="IITK Assistant",
    page_icon="favicon.ico",
    layout="centered"
)

# Apply dark theme
st.markdown(
    """
    <style>
        body {
            color: #f1f1f1;
            background-color: #111;
        }
        .stTextInput > div > div > input {
            color: white;
        }
        .reportview-container .markdown-text-container {
            color: #f1f1f1;
        }
        .stMarkdown {
            color: #f1f1f1;
        }
        footer {
            visibility: hidden;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Header with logo
st.image("logo.png", width=140)
st.markdown("<h1 style='color:#fafafa;'>IITK Assistant – Ask Anything</h1>", unsafe_allow_html=True)
st.caption("Explore information about IIT Kanpur using an AI-powered chatbot.")

# Load models and data
embedder = SentenceTransformer('all-MiniLM-L6-v2')
generator = pipeline("text2text-generation", model="google/flan-t5-base")

with open("rag_chunks.pkl", "rb") as f:
    chunks = pickle.load(f)
with open("rag_embeddings.pkl", "rb") as f:
    embeddings = pickle.load(f)

# Answering function
def answer_question(question, top_k=2):
    question_embedding = embedder.encode(question, convert_to_tensor=True)
    hits = util.semantic_search(question_embedding, embeddings, top_k=top_k)[0]
    top_chunks = [chunks[hit['corpus_id']] for hit in hits]
    context = " ".join(top_chunks)
    context_trimmed = " ".join(context.split()[:400])
    score = hits[0]['score']

    # Confidence label
    if score >= 0.7:
        label = "High Confidence"
    elif score >= 0.5:
        label = "Moderate Confidence"
    else:
        label = "Low Confidence"

    # Generate prompt
    prompt = f"""Answer the question briefly in 4-5 sentences using the context below.

Question: {question}

Context: {context_trimmed}"""

    result = generator(prompt, max_new_tokens=120)[0]['generated_text']
    return result, context_trimmed, round(score, 3), label

# Session state for chat history
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []

# Question input
question = st.text_input("💬 Enter your question:")

if question:
    with st.spinner("Generating answer..."):
        answer, context, score, confidence = answer_question(question)
        st.session_state.qa_history.append({
            "question": question,
            "answer": answer,
            "score": score,
            "confidence": confidence,
            "context": context
        })

# Show latest answer
if st.session_state.qa_history:
    latest = st.session_state.qa_history[-1]
    st.markdown("---")
    st.markdown("### Answer")
    st.write(latest["answer"])
    st.markdown(f"**Confidence:** {latest['confidence']}  (`{latest['score']}`)")

    with st.expander(" View Context Used"):
        st.write(latest["context"])

# Show previous history (last 3 questions)
if len(st.session_state.qa_history) > 1:
    st.markdown("---")
    st.subheader("🕘 Chat History (Last 3 Questions)")
    for item in st.session_state.qa_history[-2::-1][:3]:
        st.markdown(f"**Q:** {item['question']}")
        st.markdown(f"**A:** {item['answer']}")
        st.markdown(f"*Confidence:* {item['confidence']} (`{item['score']}`)")
        st.markdown("---")

# Footer / Watermark
st.markdown(
    "<p style='text-align:center; color: gray;'>Project by Mishka Gupta</p>",
    unsafe_allow_html=True
)
