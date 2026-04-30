from fastapi import FastAPI
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

# Load same embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectordb = Chroma(
    persist_directory="./vectorstore",
    embedding_function=embeddings
)

@app.post("/query")
def query_docs(req: QueryRequest):
    results = vectordb.similarity_search_with_score(
        req.query,
        k=req.top_k
    )

    output = []
    for doc, score in results:
        output.append({
            "chunk_text": doc.page_content,
            "page_number": doc.metadata.get("page"),
            "score": float(score)
        })

    return output
