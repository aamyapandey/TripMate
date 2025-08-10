# backend/embed_and_store.py
import os
import sys
import csv
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pickle
from tqdm import tqdm
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import CharacterTextSplitter
import fitz  # pymupdf
from docx import Document
from config import EMBEDDING_MODEL_NAME, CHUNK_SIZE, CHUNK_OVERLAP, FAISS_INDEX_PATH, FAISS_META_PATH

EMBEDDER = SentenceTransformer(EMBEDDING_MODEL_NAME)

def load_txt(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def load_pdf(path):
    doc = fitz.open(path)
    txt = []
    for page in doc:
        txt.append(page.get_text())
    doc.close()
    return "\n".join(txt)

def load_docx(path):
    doc = Document(path)
    return "\n".join([p.text for p in doc.paragraphs])

def load_documents(folder="data/documents"):
    docs = []
    for root, _, files in os.walk(folder):
        for f in files:
            path = os.path.join(root, f)
            ext = f.lower().split(".")[-1]
            try:
                if ext == "txt":
                    text = load_txt(path)
                    docs.append({"source": path, "text": text})
                elif ext == "pdf":
                    text = load_pdf(path)
                    docs.append({"source": path, "text": text})
                elif ext == "docx":
                    text = load_docx(path)
                    docs.append({"source": path, "text": text})
                elif ext == "csv":
                    with open(path, newline='', encoding='utf-8') as csvfile:
                        reader = csv.DictReader(csvfile)
                        for row in reader:
                            if 'query' in row and 'response' in row:
                                text = f"Q: {row['query']}\nA: {row['response']}"
                                docs.append({"source": path, "text": text})
                else:
                    continue
            except Exception as e:
                print(f"Failed to load {path}: {e}")
    return docs

def chunk_documents(docs):
    splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = []
    metadatas = []
    for doc in docs:
        pieces = splitter.split_text(doc["text"])
        for i, p in enumerate(pieces):
            chunks.append(p)
            metadatas.append({"source": doc["source"], "chunk_id": i})
    return chunks, metadatas

def create_faiss_index(chunks, metadatas, index_path=FAISS_INDEX_PATH, meta_path=FAISS_META_PATH):
    if len(chunks) == 0:
        raise ValueError("No chunks to index")
    print("Encoding chunks (this may take a while)...")
    embeddings = EMBEDDER.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
    embeddings = np.asarray(embeddings, dtype=np.float32)

    # Normalize for cosine similarity and use IndexFlatIP (inner product on normalized vectors = cosine)
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, index_path)
    with open(meta_path, "wb") as f:
        pickle.dump({"metadatas": metadatas, "chunks": chunks}, f)
    print(f"Saved index to {index_path} and metadata to {meta_path}")

if __name__ == "__main__":
    docs = load_documents()
    chunks, metas = chunk_documents(docs)
    create_faiss_index(chunks, metas)
