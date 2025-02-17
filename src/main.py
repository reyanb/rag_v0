import os
import json
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from get_index import chunk_text, summarize_text

def load_text(file_path):
    """Load text from the given file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def save_indexed_documents(indexed_documents, output_path):
    """Save the indexed documents as a JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(indexed_documents, f, ensure_ascii=False, indent=4)
    
def get_index():
    data_file = os.path.join("data", "data_1.txt")
    output_file = os.path.join("outputs", "indexed_documents.json")
    text = load_text(data_file)
    chunks = chunk_text(text)
    print(f"✅ Total Chunks: {len(chunks)}")
    indexed_documents = []
    for idx, chunk in enumerate(chunks):
        print(f"Processing chunk {idx + 1}/{len(chunks)}...")
        summary = summarize_text(chunk)
        indexed_documents.append({
            "id": idx,
            "chunk": chunk,
            "summary": summary
        })
    save_indexed_documents(indexed_documents, output_file)
    print(f"✅ Summarization Complete! Indexed documents saved to '{output_file}'.")

embedder = SentenceTransformer("dangvantuan/french-document-embedding", trust_remote_code=True)
INDEX_FILE = os.path.join("outputs", "indexed_documents.json")

def load_indexed_documents():
    """Loads the indexed documents from the output file."""
    with open(INDEX_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def initialize_embeddings():
    documents = load_indexed_documents()
    summarized_chunks = [doc["summary"] for doc in documents if "summary" in doc]
    print("🔄 Computing embeddings for indexed summaries...")
    chunk_embeddings = embedder.encode(summarized_chunks, show_progress_bar=True)
    return summarized_chunks, chunk_embeddings

if os.path.exists(INDEX_FILE):
    summarized_chunks, chunk_embeddings = initialize_embeddings()
else:
    summarized_chunks, chunk_embeddings = [], []

def retrieve_relevant_chunks(query, top_k=7):
    """
    Finds the most relevant text chunks based on a user's query.
    """
    #if not chunk_embeddings:
        #print("⚠️ No embeddings found. Please build the index first.")
        #return []
    query_embedding = embedder.encode([query])
    similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    relevant_chunks = [summarized_chunks[i] for i in top_indices]
    return relevant_chunks

VLLM_API_URL = "http://localhost:8000/v1/chat/completions"

def generate_answer(query, retrieved_text):
    """
    Generates an answer using your language model via a local vLLM API.
    The model uses the retrieved legal context to answer the user's query.
    """
    context = "\n\n".join(retrieved_text)
    prompt = (
        f"Voici un extrait de texte juridique :\n{context}\n\n"
        f"Question : {query}\n\n"
        "Répondez en français avec précision en utilisant uniquement les informations fournies."
    )
    
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
    
def main():
    # Check if the index file exists; if not, build the index.
    if not os.path.exists(INDEX_FILE):
        print("ℹ️ Index file not found. Building index...")
        get_index()
        # Reload embeddings after indexing
        global summarized_chunks, chunk_embeddings
        summarized_chunks, chunk_embeddings = initialize_embeddings()
    
    # Ask the user for a query
    query = input("\n💬 Veuillez entrer votre question : ")
    
    # Retrieve the relevant summaries/chunks from the index
    retrieved_chunks = retrieve_relevant_chunks(query)
    if not retrieved_chunks:
        print("⚠️ Aucun extrait pertinent n'a été trouvé.")
        return
    
    # Generate an answer using the retrieved context
    print("\n⏳ Génération de la réponse...")
    answer = generate_answer(query, retrieved_chunks)
    print("\n📢 Réponse :\n")
    print(answer)

if __name__ == '__main__':
    main()

