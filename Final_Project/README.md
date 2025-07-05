# IITK Assistant : Chatbot using RAG Model and FLAN-T5
Project by **Mishka Gupta** -PulpNet Final Project 2025.

This is a chatbot that can answer questions about IIT Kanpur using data from IITK CSE department website, ICS, and Vox Populi websites. It uses real AI models to retrieve and generate answers.

## Features
- Data scraped from IITK websites
- Sentence embeddings using MiniLM
- RAG based QA using FLAN-T5 model
- Confidence score + source context
- Streamlit UI with personalised logo, application logo, history, and footer

## Instructions to Run

1. Make sure Python is installed (version 3.10+)
2. Install the packages:pip install -r requirements.txt
3. Run the chatbot: streamlit run app.py


## Folder Contains:

- `app.py` - main chatbot code
- `logo.png` - logo image
- `favicon.ico` - browser tab icon
- `rag_chunks.pkl`, `rag_embeddings.pkl` - preprocessed data
- `requirements.txt` - packages needed
- `README.md` -this file





