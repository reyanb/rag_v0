from transformers import AutoTokenizer
import requests
import json

VLLM_API_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "mistralai/Mistral-Small-24B-Instruct-2501"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def chunk_text(text, chunk_size=512):
    """
    Tokenize and break the text into chunks.
    Each chunk is a string of tokens.
    """
    tokens = tokenizer(text, return_tensors="pt")["input_ids"][0].tolist()
    chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
    return [" ".join(tokenizer.convert_ids_to_tokens(chunk)) for chunk in chunks]

def summarize_text(chunk):
    """
    Sends a request to the local vLLM server for summarization.
    Returns the summary text for the given chunk.
    """
    prompt = f"Résume ce texte juridique en français:\n{chunk}"
    data = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 256
    }
    
    response = requests.post(
        VLLM_API_URL,
        headers={"Content-Type": "application/json"},
        data=json.dumps(data)
    )
    
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"].strip()
    else:
        print(f"❌ Error: {response.status_code} - {response.text}")
        return ""
