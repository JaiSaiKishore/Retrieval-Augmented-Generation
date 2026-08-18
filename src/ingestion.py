import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

# Configuration
DATA_PATH = "data"
DB_PATH = "vector_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def ingest_documents():
    # 1. Load PDF
    print("📂 Loading PDFs from 'data' folder...")
    loader = PyPDFDirectoryLoader(DATA_PATH)
    docs = loader.load()
    
    if not docs:
        print("❌ No PDFs found! Please put a PDF in the 'data' folder.")
        return

    print(f"✅ Loaded {len(docs)} pages.")

    # 2. Chunking (Breaking text into small pieces)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(docs)
    print(f"🧩 Split documents into {len(chunks)} text chunks.")

    # 3. Embeddings (Running on your RTX 3050)
    print("🚀 Initializing GPU Embeddings (this runs locally)...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cuda'} # Forces GPU usage
    )

    # 4. Save to Disk
    print("💾 Saving to ChromaDB...")
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH
    )
    print("🎉 Success! Database created in 'vector_db' folder.")

if __name__ == "__main__":
    ingest_documents()