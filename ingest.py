import argparse

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def ingest(file_path: str):

    # -----------------------------------------
    # 1. Load PDF
    # -----------------------------------------

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    print(f"Loaded {len(documents)} pages.")


    # -----------------------------------------
    # 2. Split documents
    # -----------------------------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks.")


    # -----------------------------------------
    # 3. Create embeddings
    # -----------------------------------------

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


    # -----------------------------------------
    # 4. Store vectors in Chroma
    # -----------------------------------------

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./vectorstore"
    )

    print("Ingestion complete.")
    print("Vector database saved in ./vectorstore")


# -----------------------------------------
# Command-line execution
# -----------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Ingest a PDF into a Chroma vector database."
    )

    parser.add_argument(
        "--file",
        required=True,
        help="Path to the PDF file"
    )

    args = parser.parse_args()

    ingest(args.file)
