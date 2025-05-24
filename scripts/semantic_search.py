import json
import faiss
from sentence_transformers import SentenceTransformer


VECTOR_STORE_DIR = '../vector_store/'
INDEX_FILE = f"{VECTOR_STORE_DIR}/index.faiss"
DOCS_FILE = f"{VECTOR_STORE_DIR}/METADATA.json"
EMBED_MODEL = 'sentence-transformers/paraphrase-MiniLM-L6-v2'

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

    # Fetch corresponding docs
    results = []
    for score, idx in zip(scores[0], indices[0]):
        doc = documents[idx]
        doc['score'] = float(score)
        results.append(doc)
    
    return results

if __name__ == '__main__':
    sample_query = "What pest-control methods are recommended for paddy in Tamil Nadu?"
    top_k_results = semantic_search(sample_query, top_k=3)

    print("Top Results:")
    for res in top_k_results:
        print(f"Score: {res['score']:.4f} | Q: {res['question']} | A: {res['answer']}")
