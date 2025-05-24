import requests
import json

OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3:1b"

def query_ollama(prompt: str, context: str = None) -> str:
    """
    Sends a prompt (with optional context) to the Ollama gemma3:1b model.
    """
    # Prepare the full prompt
    if context:
        full_prompt = f"Context:\n{context}\n\nQuestion: {prompt}\n\nAnswer:"
    else:
        full_prompt = prompt

    payload = {
        "model": MODEL_NAME,
        "prompt": full_prompt,
        "stream": False  # Get final response directly
    }

    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()
    except requests.RequestException as e:
        print(f"Error communicating with Ollama: {e}")
        return ""

if __name__ == "__main__":
    # Simple test
    sample_question = "What are the pest control methods for paddy?"
    sample_context = "Paddy farmers often face pests like stem borers and leaf folders. Recommended control measures include cultural practices, mechanical removal, and chemical spraying."

    answer = query_ollama(sample_question, context=sample_context)
    print("LLM Answer:", answer)
