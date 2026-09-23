from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
COLLECTION_NAME = "pdf_documents"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
