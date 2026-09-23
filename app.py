from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="PDF Vector Search API",
    description="Similarity search over PDF document chunks using Chroma.",
    version="1.0.0"
)


# --------------------------------------------------
# Request model
# --------------------------------------------------

class QueryRequest(BaseModel):

    query: str = Field(
        ...,
        min_length=1,
        description="Search query"
    )

    top_k: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Number of results to return"
    )


# --------------------------------------------------
# Load embedding model
# --------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# --------------------------------------------------
# Load Chroma database
# --------------------------------------------------

vectordb = Chroma(
    persist_directory="./vectorstore",
    embedding_function=embeddings
)


# --------------------------------------------------
# Query endpoint
# --------------------------------------------------

@app.post("/query")
def query_docs(req: QueryRequest):

    try:

        results = vectordb.similarity_search_with_score(
            req.query,
            k=req.top_k
        )

        output = []

        for doc, score in results:

            page = doc.metadata.get("page")

            # PyPDFLoader normally uses zero-based page indexing.
            # Convert to human-readable page numbering.
            page_number = page + 1 if isinstance(page, int) else page

            output.append({
                "chunk_text": doc.page_content,
                "page_number": page_number,
                "score": float(score)
            })

        return {
            "query": req.query,
            "top_k": req.top_k,
            "results": output
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Similarity search failed: {str(e)}"
        )
