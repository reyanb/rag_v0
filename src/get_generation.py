import requests
import json
VLLM_API_URL = "http://localhost:8000/v1/chat/completions"

def generate_answer(query, retrieved_text):
    """Generates an answer using Mistral 24B with retrieved legal context via local vLLM API."""
    context = "\n\n".join(retrieved_text)
    prompt = f"Voici un extrait de texte juridique :\n{context}\n\nQuestion : {query}\n\nRépondez en français avec précision en utilisant uniquement les informations fournies."
    data = {
        "model": "mistralai/Mistral-Small-24B-Instruct-2501", 
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 256  # Limit response length
    }
    response = requests.post(VLLM_API_URL, headers={"Content-Type": "application/json"}, data=json.dumps(data))
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"].strip()
    else:
        print(f"❌ Error: {response.status_code} - {response.text}")
        return "⚠️ Erreur lors de la génération de la réponse."