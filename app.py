from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from config import (
    VECTORSTORE_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
)


app = FastAPI(
    title="PDF Vector Search API",
    description="Similarity search over PDF document chunks using Chroma.",
    version="1.0.0",
)
    version="1.0.0",
)
@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "PDF Vector Search API is running."
    }

class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        description="Text query used to search the PDF document",
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Number of relevant chunks to return",
    )


embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)

vectordb = Chroma(
    persist_directory=str(VECTORSTORE_DIR),
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
)


@app.get("/")
def root():
    return {
        "message": "PDF Vector Search API is running.",
        "docs": "/docs",
        "endpoint": "POST /query",
    }


@app.post("/query")
def query_docs(req: QueryRequest):
    query = req.query.strip()

    if not query:
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty or whitespace only.",
        )

    try:
        results = vectordb.similarity_search_with_score(
            query,
            k=req.top_k,
        )

        output = []

        for doc, score in results:
            page = doc.metadata.get("page")

            if isinstance(page, int):
                page_number = page + 1
            else:
                page_number = page

            output.append(
                {
                    "chunk_text": doc.page_content,
                    "page_number": page_number,
                    "score": float(score),
                }
            )

        return {
            "query": query,
            "top_k": req.top_k,
            "results": output,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Similarity search failed: {exc}",
        ) from exc
