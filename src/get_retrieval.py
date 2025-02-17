from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

embedder = SentenceTransformer("dangvantuan/french-document-embedding", trust_remote_code=True)
input_file = "outputs/indexed_documents.json"
with open(input_file, "r", encoding="utf-8") as f:
    documents = json.load(f)
summarized_chunks = [doc["summary"] for doc in documents if "summary" in doc]
chunk_embeddings = embedder.encode(summarized_chunks, show_progress_bar=True)

def retrieve_relevant_chunks(query, top_k=7):
    """Finds the most relevant text chunks based on a user's question."""
    query_embedding = embedder.encode([query])
    similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    relevant_chunks = [summarized_chunks[i] for i in top_indices]
    
    return relevant_chunks


