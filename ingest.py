import argparse
import shutil
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from config import COLLECTION_NAME, EMBEDDING_MODEL, VECTORSTORE_DIR


def ingest(file_path: str) -> None:
    pdf_path = Path(file_path).expanduser().resolve()

    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    print(f"Loading PDF: {pdf_path}")

    # 1. Load PDF
    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()

    if not documents:
        raise ValueError("No pages were loaded from the PDF.")

    print(f"Loaded {len(documents)} pages.")

    # 2. Split documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )
    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("No text chunks were created from the PDF.")

    print(f"Created {len(chunks)} chunks.")

    # 3. Create embeddings
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # 4. Rebuild the local vector database to avoid duplicate chunks
    if VECTORSTORE_DIR.exists():
        print("Removing existing vector database...")
        shutil.rmtree(VECTORSTORE_DIR)

    # 5. Store vectors in ChromaDB
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(VECTORSTORE_DIR),
        collection_name=COLLECTION_NAME,
    )

    print("Ingestion complete.")
    print(f"Vector database saved in: {VECTORSTORE_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest a PDF into a ChromaDB vector database."
    )
    parser.add_argument(
        "--file",
        required=True,
        help="Path to the PDF file",
    )
    args = parser.parse_args()

    ingest(args.file)
