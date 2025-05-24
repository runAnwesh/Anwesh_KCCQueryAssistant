import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from ollama_client import query_ollama
import requests  # For fallback search


VECTOR_STORE_DIR = '../vector_store/'
INDEX_FILE = f"{VECTOR_STORE_DIR}/index.faiss"
DOCS_FILE = f"{VECTOR_STORE_DIR}/METADATA.json"
EMBED_MODEL = 'sentence-transformers/paraphrase-MiniLM-L6-v2'

# Fallback search endpoint (you can customize this to a real API)
FALLBACK_SEARCH_URL = "https://api.duckduckgo.com/?q={query}&format=json"

# Relevance threshold to use fallback search
RELEVANCE_THRESHOLD = 0.3

def load_index_and_docs():
    index = faiss.read_index(INDEX_FILE)
    with open(DOCS_FILE, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    return index, documents

def semantic_search(query, top_k=5):
    index, documents = load_index_and_docs()
    embedder = SentenceTransformer(EMBED_MODEL)

    query_embedding = embedder.encode([query], normalize_embeddings=True)
    scores, indices = index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        doc = documents[idx]
        doc['score'] = float(score)
        results.append(doc)
    return results

def fallback_search(query):
    print("Performing fallback search...")
    response = requests.get(FALLBACK_SEARCH_URL.format(query=query))
    if response.status_code == 200:
        data = response.json()
        # For example, return related topics (you can customize this!)
        related_topics = data.get("RelatedTopics", [])
        fallback_answers = [topic.get("Text", "") for topic in related_topics if topic.get("Text")]
        return fallback_answers or ["No relevant fallback information found."]
    else:
        return ["Fallback search failed."]

def main():
    # Example user query
    user_query = input("Enter your query: ")

    # Semantic search
    top_results = semantic_search(user_query, top_k=5)

    # Check if top result meets threshold
    top_score = top_results[0]['score'] if top_results else 0
    print(f"Top similarity score: {top_score:.4f}")

    if top_score >= RELEVANCE_THRESHOLD:
        # Concatenate context from top-k results
        context_chunks = []
        for res in top_results:
            chunk = f"Q: {res['question']} A: {res['answer']}"
            context_chunks.append(chunk)
        context = "\n\n".join(context_chunks)

        # Use Ollama LLM to generate answer
        answer = query_ollama(user_query, context=context)
        print("\n🟩 Local LLM Answer:\n", answer)
    else:
        # Perform fallback search
        fallback_results = fallback_search(user_query)
        print("\n🟥 Fallback Search Results:\n", "\n".join(fallback_results))

if __name__ == '__main__':
    main()
