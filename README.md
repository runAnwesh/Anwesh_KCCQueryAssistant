**An offline-capable, local-first AI application** that allows users to query agricultural advice from the **Kisan Call Center (KCC) dataset**. If no relevant local context is found, it seamlessly falls back to live Internet search.  
Built as part of the AI Engineer assignment.

---

## 🚀 Features

✅ Local LLM (DeepSeek via Ollama) for answering queries  
✅ Retrieval-Augmented Generation (RAG) pipeline using FAISS  
✅ Offline-capable vector search  
✅ Streamlit web app for user-friendly querying  
✅ Live fallback search when no local data is relevant  

---

## 📦 Repository Structure

```

.
├── app/                  # Streamlit app
│   └── main.py
├── data/                 # Raw & processed data
│   ├── raw/              # Raw KCC dataset
│   ├── processed/        # Cleaned JSON data
|   └──data_ingestion.ipynb     # Data collection notebook
├── llm/                  # Ollama LLM integration
│   └── launch\_llm.sh
├── scripts/              # Core pipeline scripts
│   ├── preprocess\_data.py
│   ├── generate\_embeddings.py
│   ├── semantic\_search.py
│   ├── ollama\_client.py
│   └── rag\_pipeline.py
├── vector\_store/         # FAISS index & document store
├── assets/               # Sample queries & assets
├── requirements.txt      # Python dependencies
└── README.md             # Project overview & instructions

````

---

## 📝 Workflow Overview

### 1️⃣ Data Ingestion
- **Source**: Public KCC dataset (`data/raw/KCC_raw_data.csv`)
- **Script**: `scripts/data_ingestion.ipynb` and `scripts/preprocess_data.py`
- **Output**: Cleaned & normalized JSON dataset in `data/processed/KCC_processed_data.json`

### 2️⃣ Local LLM Deployment
- **Model**: `deepseek` via Ollama
- **Script**: `llm/launch_llm.sh`  
- **Note**: Runs completely offline using quantized models if needed.

### 3️⃣ Embedding Generation & Vector Store
- **Embedding Model**: `sentence-transformers/paraphrase-MiniLM-L6-v2`
- **Script**: `scripts/generate_embeddings.py`
- **Output**: FAISS index and document store in `vector_store/`

### 4️⃣ Retrieval-Augmented Generation (RAG) Pipeline
- **Script**: `scripts/rag_pipeline.py`
- **Function**:
  - Semantic search for top-k context chunks
  - Local LLM response if relevant context found
  - Fallback live search (using Tavily Search API) if relevance threshold not met

### 5️⃣ User Interface (Streamlit App)
- **Script**: `app/main.py`
- **Features**:
  - Natural language queries
  - Displays structured LLM answers (highlighted)
  - Fallback answers clearly indicated

---

## 🔬 Sample Queries
- “What pest-control methods are recommended for paddy in Tamil Nadu?”
- “How to manage drought stress in groundnut cultivation?”
- “What issues do sugarcane farmers in Maharashtra commonly face?”
- (and at least 10 additional queries included in `assets/sample_queries.txt`)

---

## ⚙️ Installation & Running

1️⃣ Clone the repo:
```bash
git clone https://github.com/runAnwesh/Anwesh_KCCQueryAssistant.git
cd Anwesh_KCCQueryAssistant
````

2️⃣ Install dependencies:

```bash
pip install -r requirements.txt
```

3️⃣ Preprocess data:

```bash
python scripts/preprocess_data.py
```

4️⃣ Generate embeddings:

```bash
python scripts/generate_embeddings.py
```

5️⃣ Launch Ollama LLM:

```bash
cd llm
./launch_llm.sh
```

6️⃣ Run Streamlit app:

```bash
cd ..
streamlit run app/main.py
```

---

## 🎥 Demo Video

Check out the 3–5 min screencast covering:

* Local startup of LLM and vector store
* 3-5 queries returning KCC-based answers
* 2-3 queries falling back to live Internet search

**\[Google Drive Demo Video Link Here]**

---

## 📚 Technical Documentation

All scripts are heavily documented for clarity:

* Data ingestion & preprocessing (`scripts/preprocess_data.py`)
* Embedding generation (`scripts/generate_embeddings.py`)
* Vector store ingestion (`scripts/generate_embeddings.py`)
* RAG pipeline & LLM integration (`scripts/rag_pipeline.py`, `ollama_client.py`)
* Streamlit app (`app/main.py`)

---

## 🏁 Final Notes

✅ This project is built fully local-first with fallback search to ensure reliability
✅ Modular scripts allow easy swapping of embedding models or fallback search APIs
✅ Ready to deploy and extend for future enhancements (e.g., Bing API fallback, fine-tuned LLMs)

---

Feel free to reach out if you’d like help customizing this further, deploying on cloud, or extending features. Enjoy your fully functional KCC Query Assistant! 🚀🌾

```

---


