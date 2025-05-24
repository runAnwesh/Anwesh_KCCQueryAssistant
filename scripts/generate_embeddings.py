import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


PROCESSED_PATH = '../data/processed/kcc_cleaned.json'
VECTOR_STORE_DIR = '../vector_store/'
INDEX_FILE = os.path.join(VECTOR_STORE_DIR, 'index.faiss')
DOCS_FILE = os.path.join(VECTOR_STORE_DIR, 'documents.json')

EMBED_MODEL = 'sentence-transformers/all-mpnet-base-v2'

def load_processed_data():
    with open(PROCESSED_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def generate_embeddings(texts, model):
    return model.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)

def main():
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

    data = load_processed_data()
    texts = [f"Q: {d['question']} A: {d['answer']}" for d in data]

    model = SentenceTransformer(EMBED_MODEL)

    # Generate embeddings
    embeddings = generate_embeddings(texts, model)
    print(f"Generated {len(embeddings)} embeddings with dimension {embeddings.shape[1]}.")

    # Create FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product similarity
    index.add(embeddings)

    # Save index and documents
    faiss.write_index(index, INDEX_FILE)
    with open(DOCS_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"FAISS index and document store saved in {VECTOR_STORE_DIR}.")

if __name__ == '__main__':
    main()
