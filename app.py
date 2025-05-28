import streamlit as st
import os
import shutil
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI

# --- Config ---
os.environ["OPENAI_API_KEY"] = "Your working key"  
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

st.set_page_config(page_title="Simple RAG Chatbot")

st.title("RAG Chatbot")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload a PDF document to chat with it", type="pdf")

if uploaded_file is not None:
    # Save the uploaded file to the data directory
    file_path = os.path.join(DATA_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success(f"Uploaded: {uploaded_file.name}")

    # Load document and build index
    with st.spinner("Processing document..."):
        documents = SimpleDirectoryReader(DATA_DIR).load_data()
        index = VectorStoreIndex.from_documents(documents)
        llm = OpenAI(model="gpt-4.1-nano")
        query_engine = index.as_query_engine(llm=llm)
    st.success("Ready! Ask your question below.")

    # --- Chat UI ---
    question = st.text_input("Ask a question based on the document:")

    if question:
        with st.spinner("Thinking..."):
            response = query_engine.query(question)
        st.markdown(f"**Answer:** {response}")
else:
    st.warning("Please upload a PDF document to begin.")
