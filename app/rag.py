from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

documents = []
index = None

def load_documents(path="data/docs.txt"):
    global documents, index
    with open(path, "r", encoding="utf-8") as f:
        documents = [line.strip() for line in f.readlines() if line.strip()]

    embeddings = model.encode(documents)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

def retrieve(query, k=2):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), k)
    return [documents[i] for i in indices[0]]
