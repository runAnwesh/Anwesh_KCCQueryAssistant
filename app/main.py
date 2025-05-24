import streamlit as st
import json
import faiss
import numpy as np
import os
import sys
from dotenv import load_dotenv
load_dotenv()


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from sentence_transformers import SentenceTransformer
from scripts.ollama_client import query_ollama
import requests


VECTOR_STORE_DIR = os.path.join(project_root, 'vector_store')
INDEX_FILE = os.path.join(VECTOR_STORE_DIR, 'index.faiss')
DOCS_FILE = os.path.join(VECTOR_STORE_DIR, 'METADATA.json')
EMBED_MODEL = 'sentence-transformers/paraphrase-MiniLM-L6-v2'
RELEVANCE_THRESHOLD = 0.3

@st.cache_resource
def load_index_and_docs():
    index = faiss.read_index(INDEX_FILE)
    with open(DOCS_FILE, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    embedder = SentenceTransformer(EMBED_MODEL)
    return index, documents, embedder

def semantic_search(query, top_k=5):
    index, documents, embedder = load_index_and_docs()
    query_embedding = embedder.encode([query], normalize_embeddings=True)
    scores, indices = index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        doc = documents[idx]
        doc['score'] = float(score)
        results.append(doc)
    return results

def fallback_search(query):
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    if not TAVILY_API_KEY:
        return ["Tavily API key not found in environment variables."]
    
    url = "https://api.tavily.com/search"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {TAVILY_API_KEY}"
    }
    payload = {
        "query": query,
        "max_results": 1,
        "include_answer": True
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=20)
        response.raise_for_status()
        data = response.json()

        if "answer" in data and data["answer"]:
            return [data["answer"]]
        elif "results" in data and data["results"]:
            return [res["content"] for res in data["results"] if res.get("content")]
        else:
            return ["No fallback answers found."]
    except requests.RequestException as e:
        return [f"Error during Tavily search: {e}"]


def main():
    st.title("🌾 KCC Query Assistant")
    st.markdown("""
    This assistant helps you find agricultural advice using the Kisan Call Center dataset.
    If no relevant local answer is found, it performs a live Internet search as a fallback.
    """)

    user_query = st.text_input("Enter your agricultural query:")
    if user_query:
        with st.spinner("Searching local database..."):
            top_results = semantic_search(user_query, top_k=5)
            top_score = top_results[0]['score'] if top_results else 0

        if top_score >= RELEVANCE_THRESHOLD:
            st.success(f"✅ Found relevant local data (Relevance Score: {top_score:.4f})")

            # Prepare context for LLM
            context_chunks = [f"Q: {res['question']} A: {res['answer']}" for res in top_results]
            context = "\n\n".join(context_chunks)

            with st.spinner("Generating answer using local LLM..."):
                answer = query_ollama(user_query, context=context)
            st.markdown("### 🟩 Local LLM Answer")
            st.write(answer)

            # Show retrieved chunks
            with st.expander("🔍 Top Retrieved Contexts"):
                for res in top_results:
                    st.markdown(f"**Score**: {res['score']:.4f}\n- **Q**: {res['question']}\n- **A**: {res['answer']}")

        else:
            st.warning("⚠️ No local context found (Relevance Score: {top_score:.4f})")
            fallback_results = fallback_search(user_query)
            st.markdown("### 🟥 Fallback Internet Search Results")
            for res in fallback_results:
                st.write(f"- {res}")

if __name__ == '__main__':
    main()
