# backend/qa_pipeline.py
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL_NAME, FAISS_INDEX_PATH, FAISS_META_PATH, USE_OPENAI, OPENAI_API_KEY, USE_HF, HF_MODEL
from typing import List

EMBEDDER = SentenceTransformer(EMBEDDING_MODEL_NAME)

# load faiss index + metadata (lazy load)
_index = None
_store = None

def _load_store():
    global _index, _store
    if _index is None:
        if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(FAISS_META_PATH):
            raise FileNotFoundError("FAISS index or meta not found. Run embed_and_store.py first.")
        _index = faiss.read_index(FAISS_INDEX_PATH)
        with open(FAISS_META_PATH, "rb") as f:
            _store = pickle.load(f)
    return _index, _store

def retrieve(query: str, top_k: int = 4):
    index, store = _load_store()
    q_emb = EMBEDDER.encode([query], convert_to_numpy=True)
    q_emb = np.asarray(q_emb, dtype=np.float32)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, top_k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx == -1:
            continue
        results.append({
            "score": float(score),
            "text": store["chunks"][idx],
            "meta": store["metadatas"][idx]
        })
    return results

def _openai_generate(system_prompt: str, user_prompt: str):
    import openai
    openai.api_key = OPENAI_API_KEY
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=512,
        temperature=0.2
    )
    return resp["choices"][0]["message"]["content"].strip()

def _hf_generate(prompt: str, max_new_tokens: int = 256):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
    import torch
    tok = AutoTokenizer.from_pretrained(HF_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(HF_MODEL)
    generator = pipeline("text2text-generation", model=model, tokenizer=tok, device=0 if torch.cuda.is_available() else -1)
    out = generator(prompt, max_new_tokens=max_new_tokens, do_sample=False)
    return out[0]["generated_text"]

def generate_answer(question: str, context_chunks: List[str]):
    context = "\n\n---\n\n".join(context_chunks)
    system_prompt = "You are a helpful assistant. Use the provided context to answer the user's question. If the context does not contain the answer, say you don't know — do not hallucinate facts."
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer concisely and cite sources in the form [source_path|chunk_id] when relevant."
    if USE_OPENAI:
        return _openai_generate(system_prompt, user_prompt)
    elif USE_HF:
        prompt = system_prompt + "\n\n" + user_prompt
        return _hf_generate(prompt)
    else:
        raise RuntimeError("No generation backend configured. Set OPENAI_API_KEY or enable USE_HF.")

def get_answer(query: str, top_k: int = 4):
    hits = retrieve(query, top_k=top_k)
    contexts = [h["text"] for h in hits]
    ans = generate_answer(query, contexts)
    return {"answer": ans, "retrieved": hits}
