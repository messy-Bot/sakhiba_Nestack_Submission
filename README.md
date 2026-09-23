# sakhiba_Nestack_Submission

## PDF Vectorization and Similarity Search Pipeline

A document processing and semantic search pipeline that converts PDF documents into vector embeddings and stores them in a **Chroma vector database**. The project uses **LangGraph** to orchestrate the document ingestion workflow and **FastAPI** to provide an API for similarity-based search.

The system retrieves the most relevant text chunks from an uploaded PDF based on a user's query **without using an LLM or retrieval chain**.

---

## Project Overview

The pipeline follows these main steps:

```text
PDF Document
     │
     ▼
PyPDFLoader
     │
     ▼
Document Text
     │
     ▼
Recursive Character Text Splitter
     │
     ▼
Text Chunks
     │
     ▼
Hugging Face Embeddings
     │
     ▼
Chroma Vector Database
     │
     ▼
FastAPI
     │
     ▼
User Query
     │
     ▼
Similarity Search
     │
     ▼
Top-K Relevant Chunks
```

The project consists of two main stages:

1. **Document ingestion and vectorization**
2. **API-based similarity search**

---

## Features

* PDF document loading using PyPDF
* Automatic document chunking
* Configurable chunk size and overlap
* Sentence-transformer based embeddings
* Persistent Chroma vector database
* LangGraph-based ingestion workflow
* FastAPI REST API
* Top-K similarity search
* Page number information in search results
* Similarity/distance scores for retrieved chunks
* No LLM required
* No retrieval chain required
* Interactive API documentation through Swagger UI

---

## Technologies Used

| Technology            | Purpose                                       |
| --------------------- | --------------------------------------------- |
| Python                | Core programming language                     |
| LangChain             | Document processing and embedding integration |
| LangGraph             | Workflow orchestration                        |
| Hugging Face          | Text embedding model                          |
| Sentence Transformers | Semantic text embeddings                      |
| Chroma                | Vector database                               |
| FastAPI               | REST API                                      |
| Pydantic              | Request validation                            |
| PyPDF                 | PDF document loading                          |
| Uvicorn               | FastAPI server                                |

---

## Embedding Model

The project uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The same embedding model is used during both:

* document vectorization
* query processing

This ensures that document chunks and user queries are represented in the same vector space.

---

# Project Structure

```text
sakhiba_Nestack_Submission/
│
├── app.py
├── graph_ingest.py
├── ingest.py
├── requirements.txt
├── results.json
├── sample.pdf
├── README.md
│
└── vectorstore/
    └── Chroma database files
```

### File Description

#### `app.py`

FastAPI application that loads the Chroma vector database and provides the `/query` endpoint for similarity search.

#### `graph_ingest.py`

Main LangGraph-based ingestion workflow.

It performs:

```text
Load PDF
   ↓
Split Documents
   ↓
Generate Embeddings
   ↓
Store in Chroma
```

#### `ingest.py`

Alternative command-line ingestion script that performs the same basic PDF-to-Chroma process without LangGraph.

#### `requirements.txt`

Contains the Python dependencies required to run the project.

#### `results.json`

Contains sample output from similarity-search queries.

#### `sample.pdf`

Sample PDF document used for testing the ingestion pipeline.

#### `vectorstore/`

Persistent Chroma database containing the generated document embeddings.

---

# Installation

## 1. Clone the repository

```bash
git clone https://github.com/messy-Bot/sakhiba_Nestack_Submission.git
```

Navigate into the project:

```bash
cd sakhiba_Nestack_Submission
```

---

## 2. Create a virtual environment

On Windows:

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

On Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

# Requirements

The project uses the following packages:

```text
fastapi
uvicorn[standard]
pydantic

langchain
langchain-core
langchain-community
langchain-huggingface
langchain-text-splitters
langchain-chroma
langgraph

chromadb
pypdf
sentence-transformers
```

---

# Document Ingestion

Before using the API, the PDF must be processed and stored in the Chroma vector database.

## Option 1: LangGraph ingestion

The primary workflow uses `graph_ingest.py`.

Place your PDF in the project directory and name it:

```text
sample.pdf
```

Then run:

```bash
python graph_ingest.py
```

The workflow performs:

```text
sample.pdf
    ↓
PyPDFLoader
    ↓
Document pages
    ↓
RecursiveCharacterTextSplitter
    ↓
Text chunks
    ↓
HuggingFaceEmbeddings
    ↓
Chroma
    ↓
./vectorstore
```

Expected console output:

```text
Loaded X pages from: sample.pdf
Created X text chunks
Embeddings stored in Chroma vector database
Ingestion pipeline completed successfully.
```

`X` depends on the number of pages and amount of text in the PDF.

---

# Text Chunking

The project uses:

```python
RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
```

### Chunk Size

```text
500 characters
```

Each document is divided into smaller text chunks of approximately 500 characters.

### Chunk Overlap

```text
100 characters
```

The overlap helps preserve contextual information between neighboring chunks.

For example:

```text
Chunk 1:
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA

Chunk 2:
                    AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
```

The overlapping content can help prevent important information from being separated completely between chunks.

---

# Vectorization

Each text chunk is converted into a numerical vector using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Conceptually:

```text
Text Chunk
    ↓
Embedding Model
    ↓
Numerical Vector
    ↓
Chroma Vector Database
```

These vectors allow semantic similarity searches rather than relying only on exact keyword matching.

---

# Chroma Vector Database

The generated vectors are stored in:

```text
./vectorstore
```

The database is persistent, so the FastAPI application can load the stored vectors later without processing the PDF again.

---

# Running the API

After ingestion is complete, start the FastAPI server:

```bash
uvicorn app:app --reload
```

The API will normally be available at:

```text
http://127.0.0.1:8000
```

---

# API Documentation

FastAPI automatically provides interactive Swagger documentation.

Open:

```text
http://127.0.0.1:8000/docs
```

You can test the API directly from the Swagger interface.

---

# Query API

## Endpoint

```text
POST /query
```

### Request

```json
{
  "query": "What is the main topic of the document?",
  "top_k": 3
}
```

### Parameters

| Parameter | Type    | Required | Description                 |
| --------- | ------- | -------- | --------------------------- |
| `query`   | string  | Yes      | Text to search for          |
| `top_k`   | integer | No       | Number of results to return |

The default value of `top_k` is:

```text
3
```

The API accepts values between:

```text
1 and 20
```

---

# Example Query

```json
{
  "query": "Explain embeddings",
  "top_k": 3
}
```

---

# Example Response

```json
{
  "query": "Explain embeddings",
  "top_k": 3,
  "results": [
    {
      "chunk_text": "Embeddings are numerical representations of text...",
      "page_number": 2,
      "score": 0.78
    },
    {
      "chunk_text": "Vector representations allow semantic comparison...",
      "page_number": 3,
      "score": 0.84
    }
  ]
}
```

The actual text, page numbers, and scores depend on the PDF being searched.

---

# Search Process

When a user submits a query:

```text
User Query
    │
    ▼
Hugging Face Embedding Model
    │
    ▼
Query Vector
    │
    ▼
Chroma Vector Database
    │
    ▼
Similarity Search
    │
    ▼
Top-K Matching Chunks
    │
    ▼
FastAPI JSON Response
```

---

# Score

The API returns a `score` for each retrieved chunk.

Example:

```json
"score": 0.78
```

This value is the score returned by the vector-store similarity search.

It should **not automatically be interpreted as a percentage similarity**. Its exact interpretation depends on the distance/similarity configuration used by the vector store.

---

# Page Numbers

PDF page metadata is preserved during document loading.

The API converts the zero-based page index into a human-readable page number.

For example:

```text
Internal page index: 0
Displayed page number: 1
```

This allows users to identify where the retrieved text came from in the original PDF.

---

# LangGraph Workflow

The ingestion pipeline is implemented using LangGraph.

The workflow consists of three nodes:

```text
START
  │
  ▼
load
  │
  ▼
split
  │
  ▼
embed_store
  │
  ▼
END
```

### Load Node

```python
load_node()
```

Loads the PDF using `PyPDFLoader`.

### Split Node

```python
split_node()
```

Splits the loaded document into smaller chunks.

### Embed and Store Node

```python
embed_store_node()
```

Generates embeddings and stores the chunks in Chroma.

---

# Alternative Ingestion Script

The project also contains:

```text
ingest.py
```

This provides a simpler ingestion workflow without LangGraph.

Run it with:

```bash
python ingest.py --file sample.pdf
```

Expected output:

```text
Loaded X pages.
Created X chunks.
Ingestion complete.
Vector database saved in ./vectorstore
```

The LangGraph version is the primary workflow, while `ingest.py` provides a simpler alternative.

---

# No LLM Architecture

This project intentionally does **not** use an LLM.

It also does not use:

* RetrievalQA
* ConversationalRetrievalChain
* LLM-based answer generation
* LangChain retrieval chains

Instead, it directly performs:

```text
Query
 ↓
Embedding
 ↓
Vector Search
 ↓
Relevant Text Chunks
```

The API returns the retrieved source chunks rather than generating an answer with an LLM.

---

# Advantages

* Lightweight semantic search
* No LLM API key required
* Lower computational requirements than an LLM-based pipeline
* Persistent vector storage
* Fast retrieval of relevant document chunks
* Easy to expose through a REST API
* Modular ingestion and search components
* LangGraph provides explicit workflow orchestration

---

# Future Improvements

Possible future enhancements include:

* Support for multiple PDF files
* File upload endpoint
* Metadata filtering
* Configurable chunk size
* Configurable embedding models
* Improved error handling
* Authentication for the API
* Batch document ingestion
* Search result ranking and filtering
* Web-based frontend
* Optional LLM layer for answer generation
* Docker deployment
* Cloud-based vector database
* Automated testing and CI/CD

---

# Example End-to-End Workflow

```text
1. Place PDF in project directory
             ↓
2. Run graph_ingest.py
             ↓
3. PDF is loaded
             ↓
4. Text is split into chunks
             ↓
5. Chunks are converted into embeddings
             ↓
6. Embeddings are stored in Chroma
             ↓
7. Start FastAPI
             ↓
8. Send POST /query request
             ↓
9. Query is converted into an embedding
             ↓
10. Chroma performs similarity search
             ↓
11. Top-K relevant chunks are returned
```

---

# Sample Queries

### Query 1

```json
{
  "query": "What is the main topic of the document?",
  "top_k": 3
}
```

### Query 2

```json
{
  "query": "Explain embeddings",
  "top_k": 3
}
```

### Query 3

```json
{
  "query": "What is LangChain used for?",
  "top_k": 3
}
```

---

# Project Objective

The objective of this project is to demonstrate how unstructured PDF documents can be transformed into searchable vector representations and queried using semantic similarity.

The project focuses on the **document ingestion, embedding, vector storage, and retrieval layers** rather than LLM-based response generation.

---

# Author

**Shaik Ture Sakhiba Banu**

B.Tech – Computer Science and Engineering

GitHub: `messy-Bot`

---
## License

This project is created for educational and technical demonstration purposes.
