# main.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
import shutil
import numpy as np

from models.chunker import TextChunker
from models.embedding_model import EmbeddingModel
from models.file_loader import FileLoader
from models.llm_model import LLMModel
from vectorstore.faiss_store import FAISSStore

# --------------------------
# FastAPI app
# --------------------------
app = FastAPI()
vector_store = None
chunks = []

# --------------------------
# HTML UI
# --------------------------
HTML_UI = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>🤖 RAG Chatbot</title>
</head>
<body style="font-family:Arial; background:#f0f0f0; padding:20px;">
    <h1>🤖 RAG Chatbot</h1>
    <div>
        <h3>📂 Upload File (PDF / TXT)</h3>
        <form action="/upload" method="post" enctype="multipart/form-data">
            <input type="file" name="file">
            <button type="submit">Upload</button>
        </form>
    </div>
    <div>
        <h3>🌐 Enter URL</h3>
        <form action="/upload" method="post">
            <input type="text" name="url" placeholder="https://example.com">
            <button type="submit">Upload</button>
        </form>
    </div>
    <div>
        <h3>💬 Ask Question</h3>
        <form action="/ask" method="post">
            <input type="text" name="question" placeholder="Type your question here">
            <button type="submit">Ask</button>
        </form>
    </div>
</body>
</html>
"""

# --------------------------
# Root endpoint
# --------------------------
@app.get("/", response_class=HTMLResponse)
async def root():
    return HTML_UI

# --------------------------
# Upload endpoint
# --------------------------
@app.post("/upload")
async def upload_file(file: UploadFile = File(None), url: str = Form(None)):
    global vector_store, chunks
    text = ""

    if file:
        file_path = f"temp_{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        if file.filename.lower().endswith(".pdf"):
            loader = FileLoader(file_path=file_path)
            text = loader.load("pdf")
        elif file.filename.lower().endswith(".txt"):
            loader = FileLoader(file_path=file_path)
            text = loader.load("txt")
        else:
            return {"error": "Unsupported file type"}
    elif url:
        loader = FileLoader(url=url)
        text = loader.load("url")
    else:
        return {"error": "No file or URL provided"}

    if not text.strip():
        return {"error": "No text extracted from input"}

    # Chunking
    chunker = TextChunker(chunk_size=500, overlap=50)
    chunks = chunker.chunk_text(text)

    # Embeddings
    embedder = EmbeddingModel()
    embeddings = [embedder.get_embedding(c) for c in chunks if c.strip()]
    if not embeddings:
        return {"error": "Failed to generate embeddings"}

    # FAISS store
    dim = len(embeddings[0])
    vector_store = FAISSStore(dim)
    vector_store.add_vectors(np.array(embeddings))

    return {"message": "Input processed successfully", "chunks": len(chunks)}

# --------------------------
# Ask endpoint
# --------------------------
@app.post("/ask")
async def ask_question(question: str = Form(...)):
    global vector_store, chunks
    if vector_store is None:
        return {"error": "No document uploaded yet"}

    embedder = EmbeddingModel()
    query_vector = embedder.get_embedding(question)

    distances, indices = vector_store.search(query_vector, top_k=3)
    retrieved_chunks = [chunks[i] for i in indices[0]]
    context = "\n".join(retrieved_chunks)

    llm = LLMModel()
    answer = llm.generate_response(question, context)

    return {"answer": answer}

# --------------------------
# Run the app:
# uvicorn main:app --reload
# --------------------------