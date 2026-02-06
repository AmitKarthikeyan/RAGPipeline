import os
import time
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from pinecone import Pinecone, ServerlessSpec

# -------------------
# Config
# -------------------
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Set these to match your Pinecone project (change if your dashboard shows a different region)
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "pdf-index")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")
EMBED_DIM = 384  # all-MiniLM-L6-v2 outputs 384-dim vectors

genai.configure(api_key=GOOGLE_API_KEY)


# -------------------
# Helpers
# -------------------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += (page.extract_text() or "")
    return text


def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000,
    )
    return splitter.split_text(text)


def ensure_pinecone_index(pc: Pinecone, index_name: str, dimension: int):
    """Create Pinecone index if it doesn't exist and wait until ready."""
    existing = {i["name"] for i in pc.list_indexes()}

    if index_name not in existing:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
        st.info(f"Created Pinecone index: {index_name}")

    # Wait for readiness (important right after create)
    for _ in range(60):
        desc = pc.describe_index(index_name)
        status = getattr(desc, "status", None)
        if status and getattr(status, "ready", False):
            return
        time.sleep(1)


def get_vector_store(text_chunks):
    if not PINECONE_API_KEY:
        st.error("Missing PINECONE_API_KEY in your environment/.env.")
        return None

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

    pc = Pinecone(api_key=PINECONE_API_KEY)
    ensure_pinecone_index(pc, INDEX_NAME, dimension=EMBED_DIM)

    index = pc.Index(INDEX_NAME)

    # Upsert texts into Pinecone
    vector_store = PineconeVectorStore.from_texts(
        texts=text_chunks,
        embedding=embeddings,
        index_name=INDEX_NAME,
        pinecone_api_key=PINECONE_API_KEY,
    )
    return vector_store


def answer_with_context(docs, user_question):
    prompt_template = """
Answer the question as detailed as possible from the provided context. If the answer is not in the context,
say "answer is not available in the context". Do not guess.

Context:
{context}

Question:
{question}

Answer:
"""
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    context = "\n\n".join([d.page_content for d in docs])
    message = prompt.format(context=context, question=user_question)
    return model.invoke(message)


def user_input(user_question):
    if not PINECONE_API_KEY:
        st.error("Missing PINECONE_API_KEY in your environment/.env.")
        return

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # If the index doesn't exist yet, user hasn't processed PDFs
    existing = {i["name"] for i in pc.list_indexes()}
    if INDEX_NAME not in existing:
        st.error("No Pinecone index found yet. Upload PDFs and click Process first.")
        return

    index = pc.Index(INDEX_NAME)
    vector_store = PineconeVectorStore(index=index, embedding=embeddings, index_name=INDEX_NAME)
    docs = vector_store.similarity_search(user_question, k=4)
    result = answer_with_context(docs, user_question)
    st.write("Reply:", result.content)


# -------------------
# Streamlit UI
# -------------------
def main():
    st.set_page_config(page_title="Chat With Multiple PDF", page_icon="📄")
    st.header("Chat With Multiple PDF using Gemini")

    user_question = st.text_input("Ask your question about the PDF:")
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'",
            accept_multiple_files=True
        )

        if st.button("Process"):
            if not pdf_docs:
                st.error("Please upload at least one PDF.")
                return

            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vs = get_vector_store(text_chunks)
                if vs is not None:
                    st.success("Processing completed!")


if __name__ == "__main__":
    main()
