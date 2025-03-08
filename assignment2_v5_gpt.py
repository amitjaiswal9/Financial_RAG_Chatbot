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
SLM_MODEL = "google/flan-t5-small"
BLACKLISTED_WORDS = ["politics", "religion", "offensive"]
SIMILARITY_THRESHOLD = 0.6  # Adjusted similarity threshold for hallucination check

# Step 1: Data Collection & Preprocessing
# Use local files instead of downloading
def get_local_pdfs():
    return ["Financial_Year_2023_GEN.pdf"]

# Extract text from PDFs
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

# Step 2: Basic RAG - Text Chunking & Embedding
def chunk_text(text, chunk_size=512):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# Embed chunks
def embed_chunks(chunks, model):
    return model.encode(chunks, convert_to_numpy=True)

# Step 3: Advanced RAG - BM25 & FAISS
def build_bm25_index(chunks):
    tokenized_corpus = [chunk.split() for chunk in chunks]
    return BM25Okapi(tokenized_corpus)

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

# Step 4: Query Processing & Retrieval
def retrieve_documents(query, bm25, faiss_index, embed_model, chunks, top_k=3):
    # BM25 retrieval
    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_top_k = np.argsort(bm25_scores)[-top_k:][::-1]
    
    # Vector search
    query_embedding = embed_model.encode([query])
    _, faiss_top_k = faiss_index.search(query_embedding, top_k)
    faiss_top_k = faiss_top_k[0]
    
    # Merge results
    retrieved_chunks = list(set([chunks[i] for i in bm25_top_k] + [chunks[i] for i in faiss_top_k]))
    return "\n".join(retrieved_chunks)

# Step 5: Response Generation (SLM)
def generate_response(context, query, model):
    input_text = f"Context: {context} \n Question: {query}"
    response = model(input_text, max_length=200, truncation=True)[0]['generated_text']
    return response

# Step 6: Guardrails
def validate_query(query):
    if any(word in query.lower() for word in BLACKLISTED_WORDS):
        return False
    if not re.match(r'^[a-zA-Z0-9 ?!.]+$', query):  # Basic regex check
        return False
    return True

def check_hallucination(query, response, embed_model):
    query_embedding = embed_model.encode([query], convert_to_tensor=True)
    response_embedding = embed_model.encode([response], convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(query_embedding, response_embedding).item()
    return similarity >= SIMILARITY_THRESHOLD, similarity

# Step 7: UI (Streamlit)
def main():
    st.title("Financial RAG Chatbot")
    query = st.text_input("Enter your financial question:")
    
    if st.button("Search"):
        if validate_query(query):
            context = retrieve_documents(query, bm25, faiss_index, embed_model, chunks)
            response = generate_response(context, query, slm_model)
            
            # Guardrail: Output Filtering
            is_valid, confidence_score = check_hallucination(query, response, embed_model)
            if is_valid:
                confidence = f"High ({confidence_score:.2f})"
            else:
                confidence = f"Low ({confidence_score:.2f}) - Response may be unreliable."
                response = "This response may not be accurate. Please verify from official sources."
            
            st.markdown(f"**Retrieved Context:**\n{context}")
            st.markdown(f"**Answer:** {response}")
            st.markdown(f"**Confidence Score:** {confidence}")
        else:
            st.write("Invalid query. Please ask a relevant financial question.")

if __name__ == "__main__":
    pdf_paths = get_local_pdfs()  # Use local PDFs
    texts = extract_text_from_pdfs(pdf_paths)
    chunks = [chunk for text in texts for chunk in chunk_text(text)]
    
    embed_model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = embed_chunks(chunks, embed_model)
    faiss_index = build_faiss_index(embeddings)
    bm25 = build_bm25_index(chunks)
    
    slm_model = pipeline("text2text-generation", model=SLM_MODEL)
    main()
