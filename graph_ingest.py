from langgraph.graph import StateGraph
from typing import TypedDict
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

class State(TypedDict):
    file_path: str
    documents: list
    chunks: list

def load_node(state):
    loader = PyPDFLoader(state["file_path"])
    return {"documents": loader.load()}

def split_node(state):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return {"chunks": splitter.split_documents(state["documents"])}

def embed_store_node(state):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    Chroma.from_documents(
        documents=state["chunks"],
        embedding=embeddings,
        persist_directory="./vectorstore"
    ).persist()

    return {}

builder = StateGraph(State)

builder.add_node("load", load_node)
builder.add_node("split", split_node)
builder.add_node("embed_store", embed_store_node)

builder.set_entry_point("load")
builder.add_edge("load", "split")
builder.add_edge("split", "embed_store")

graph = builder.compile()

if __name__ == "__main__":
    graph.invoke({"file_path": "sample.pdf"})
