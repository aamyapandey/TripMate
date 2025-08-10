# config.py
import os

# Embedding model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Chunking
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50




# OpenAI settings (disabled)
USE_OPENAI = False
OPENAI_API_KEY = ""

# If not using OpenAI, set a HuggingFace model id here for local generation
# WARNING: many HF models need GPU; prefer OpenAI if you don't have GPU.
USE_HF = True
HF_MODEL = "google/flan-t5-large"  # example; change as needed

# Filenames for vector store
FAISS_INDEX_PATH = "faiss.index"
FAISS_META_PATH = "faiss_meta.pkl"
