from pathlib import Path
import shutil
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from config import COLLECTION_NAME, EMBEDDING_MODEL, VECTORSTORE_DIR


class State(TypedDict, total=False):
    file_path: str
    documents: list
    chunks: list


def load_node(state: State):
    file_path = Path(state["file_path"]).expanduser().resolve()

    if not file_path.is_file():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    loader = PyPDFLoader(str(file_path))
    documents = loader.load()

    if not documents:
        raise ValueError(f"No pages were loaded from PDF: {file_path}")

    print(f"Loaded {len(documents)} pages from: {file_path}")
    return {"documents": documents}


def split_node(state: State):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )

    chunks = splitter.split_documents(state["documents"])

    if not chunks:
        raise ValueError("No text chunks were created from the PDF.")

    print(f"Created {len(chunks)} text chunks")
    return {"chunks": chunks}


def embed_store_node(state: State):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Rebuild the local vector database to avoid duplicate chunks.
    if VECTORSTORE_DIR.exists():
        shutil.rmtree(VECTORSTORE_DIR)

    Chroma.from_documents(
        documents=state["chunks"],
        embedding=embeddings,
        persist_directory=str(VECTORSTORE_DIR),
        collection_name=COLLECTION_NAME,
    )

    print("Embeddings stored in ChromaDB")
    return {}


builder = StateGraph(State)

builder.add_node("load", load_node)
builder.add_node("split", split_node)
builder.add_node("embed_store", embed_store_node)

builder.add_edge(START, "load")
builder.add_edge("load", "split")
builder.add_edge("split", "embed_store")
builder.add_edge("embed_store", END)

graph = builder.compile()


if __name__ == "__main__":
    sample_pdf = Path(__file__).resolve().parent / "sample.pdf"

    graph.invoke({"file_path": str(sample_pdf)})

    print("LangGraph ingestion pipeline completed successfully.")
