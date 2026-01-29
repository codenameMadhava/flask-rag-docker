from flask import Flask, request, jsonify
from rag import load_documents, retrieve

app = Flask(__name__)
load_documents()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/query")
def query():
    data = request.get_json(force=True)
    question = data.get("question", "")

    context = retrieve(question, k=2)
    return jsonify({
        "question": question,
        "context": context,
        "answer": f"Answer based on context: {context}"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
