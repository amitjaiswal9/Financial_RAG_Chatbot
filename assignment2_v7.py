import os
import pdfplumber
import numpy as np
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
from transformers import pipeline
import re

# Constants
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SLM_MODEL = "google/flan-t5-large"
BLACKLISTED_WORDS = ["politics", "religion", "offensive"]
SIMILARITY_THRESHOLD = 0.3
CONFIDENCE_THRESHOLD = 0.2  # Lowered threshold for better response acceptance
FINANCE_KEYWORDS = ["revenue", "profit", "loss", "investment", "stock", "dividend", "earnings", "financial", "share", "price", "assets", "liabilities", "cash flow", "equity", "debt", "capital", "expenses", "interest", "tax", "balance sheet", "income statement", "net income", "operating income", "gross profit", "return on investment", "market cap", "valuation", "leverage", "credit", "bonds", "amortization", "depreciation", "liquidity", "working capital", "EBITDA", "fiscal year", "portfolio", "derivatives", "hedging", "macroeconomics", "inflation", "GDP", "risk management"]

@st.cache_resource
def get_local_pdfs():
    return ["TCS-2022-2023.pdf","TCS-2023-2024.pdf",]

@st.cache_resource
def extract_text_from_pdfs(pdf_paths):
    text_data = []
    for pdf_path in pdf_paths:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = "\n".join([page.extract_text() or "" for page in pdf.pages])
                text_data.append(text)
        except Exception as e:
            print(f"Error reading {pdf_path}: {e}")
    return text_data

def chunk_text(text, chunk_size=256):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

@st.cache_resource
def load_embed_model():
    return SentenceTransformer(EMBEDDING_MODEL)

@st.cache_resource
def embed_chunks(chunks, _model):
    return _model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)

@st.cache_resource
def build_bm25_index(chunks):
    tokenized_corpus = [chunk.split() for chunk in chunks]
    return BM25Okapi(tokenized_corpus)

@st.cache_resource
def build_faiss_index(embeddings):
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index

def retrieve_documents(query, bm25, faiss_index, embed_model, chunks, top_k=10):
    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_top_k = np.argsort(bm25_scores)[-top_k:][::-1]
    
    query_embedding = embed_model.encode([query], normalize_embeddings=True)
    _, faiss_top_k = faiss_index.search(query_embedding, top_k)
    faiss_top_k = faiss_top_k[0]
    
    retrieved_chunks = list(set([chunks[i] for i in bm25_top_k] + [chunks[i] for i in faiss_top_k]))
    return "\n".join(retrieved_chunks) if retrieved_chunks else "No relevant data found."

@st.cache_resource
def load_slm_model():
    try:
        return pipeline("text2text-generation", model=SLM_MODEL)
    except Exception as e:
        print(f"Error loading SLM model: {e}")
        return None

def generate_response(context, query, model):
    if not context.strip():
        return "I couldn't find relevant data for your question. Please refine your query."
    input_text = f"Context: {context}\n\nQuestion: {query}\n\nAnswer strictly using the provided data. If unsure, reply 'I don't know'."
    response = model(input_text, max_length=300, truncation=True)[0]['generated_text']
    return response

def validate_query(query):
    if not query.strip():
        return False  # Reject empty queries
    if any(word in query.lower() for word in BLACKLISTED_WORDS):
        return False  # Reject blacklisted content
    if not any(keyword in query.lower() for keyword in FINANCE_KEYWORDS):
        return False  # Encourage financial queries
    return True

def check_hallucination(response, context, embed_model):
    response_embedding = embed_model.encode([response], normalize_embeddings=True)
    context_embedding = embed_model.encode([context], normalize_embeddings=True)
    similarity = util.pytorch_cos_sim(response_embedding, context_embedding).item()
    return similarity >= SIMILARITY_THRESHOLD, similarity

def main():
    st.title("Financial RAG Chatbot")
    query = st.text_input("Enter your financial question:")
    
    if st.button("Search"):
        validation_result = validate_query(query)
        if not validation_result:
            st.write("Invalid query. Please ask a relevant financial question.")
            return
        
        context = retrieve_documents(query, bm25, faiss_index, embed_model, chunks)
        response = generate_response(context, query, slm_model)
        
        is_valid, confidence_score = check_hallucination(response, context, embed_model)
        if confidence_score < CONFIDENCE_THRESHOLD:
            response = "I don't have enough reliable data to answer this accurately."
            confidence = f"Low ({confidence_score:.2f})"
        else:
            confidence = f"High ({confidence_score:.2f})" if is_valid else f"Medium ({confidence_score:.2f})"
        
        st.markdown(f"**Answer:** {response}")
        st.markdown(f"**Confidence Score:** {confidence}")

if __name__ == "__main__":
    pdf_paths = get_local_pdfs()
    texts = extract_text_from_pdfs(pdf_paths)
    chunks = [chunk for text in texts for chunk in chunk_text(text)]
    
    embed_model = load_embed_model()
    embeddings = embed_chunks(chunks, _model=embed_model)
    faiss_index = build_faiss_index(embeddings)
    bm25 = build_bm25_index(chunks)
    
    slm_model = load_slm_model()
    main()
