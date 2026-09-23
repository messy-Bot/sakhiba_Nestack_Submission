from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# --------------------------------------------------
# State definition
# --------------------------------------------------

class State(TypedDict, total=False):
    file_path: str
    documents: list
    chunks: list


# --------------------------------------------------
# Node 1: Load PDF
# --------------------------------------------------

def load_node(state: State):
    file_path = state["file_path"]

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    print(f"Loaded {len(documents)} pages from: {file_path}")

    return {
        "documents": documents
    }


# --------------------------------------------------
# Node 2: Split documents into chunks
# --------------------------------------------------

def split_node(state: State):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(
        state["documents"]
    )

    print(f"Created {len(chunks)} text chunks")

    return {
        "chunks": chunks
    }


# --------------------------------------------------
# Node 3: Create embeddings and store in Chroma
# --------------------------------------------------

def embed_store_node(state: State):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    Chroma.from_documents(
        documents=state["chunks"],
        embedding=embeddings,
        persist_directory="./vectorstore"
    )

    print("Embeddings stored in Chroma vector database")

    return {}


# --------------------------------------------------
# Build LangGraph workflow
# --------------------------------------------------

builder = StateGraph(State)

builder.add_node("load", load_node)
builder.add_node("split", split_node)
builder.add_node("embed_store", embed_store_node)

builder.add_edge(START, "load")
builder.add_edge("load", "split")
builder.add_edge("split", "embed_store")
builder.add_edge("embed_store", END)

graph = builder.compile()


# --------------------------------------------------
# Run the ingestion pipeline
# --------------------------------------------------

if __name__ == "__main__":

    graph.invoke({
        "file_path": "sample.pdf"
    })

    print("Ingestion pipeline completed successfully.")
