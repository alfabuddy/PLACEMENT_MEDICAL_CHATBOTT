import os
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from src.helper import load_pdf_files, text_split, get_embeddings

# Load env
load_dotenv()

INDEX_NAME = "medical-chatbot-384"

if __name__ == "__main__":
    print("📄 Loading PDFs...")
    documents = load_pdf_files("data/")

    print("✂️ Splitting text...")
    text_chunks = text_split(documents)

    print("🧠 Loading FastEmbed embeddings...")
    embeddings = get_embeddings()

    print(f"📥 Uploading {len(text_chunks)} chunks to Pinecone...")
    PineconeVectorStore.from_documents(
        documents=text_chunks,
        embedding=embeddings,
        index_name=INDEX_NAME
    )

    print("✅ Ingestion completed successfully")
