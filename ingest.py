"""
ingest.py
----------
Ingiere todos los PDFs en ./docs, los trocea, genera
embeddings con BAAI/bge-base-en-v1.5 (768 dims) y
los guarda en la colección 'manual_bge_base'.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.pgvector import PGVector

# 1) Entorno
load_dotenv()
PG_CONN = os.environ["PG_CONN"]

PDF_DIR = Path("docs")
EMBED_MODEL = "intfloat/multilingual-e5-base"

embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu"},
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=80,
    separators=["\n\n", "\n", " ", ""],
)

def ingest_pdf(pdf_path: Path):
    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()
    for d in docs:
        d.metadata["source"] = pdf_path.name
    return splitter.split_documents(docs)

def main():
    print("🔍 PDFs en", PDF_DIR.resolve())
    pdf_files = list(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError("⚠️  No hay PDFs en ./docs")

    chunks = []
    for pdf in pdf_files:
        print(f"📄 {pdf.name}")
        chunks.extend(ingest_pdf(pdf))

    print(f"✂️  Segmentos: {len(chunks)}")

    PGVector.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="manual_e5_multi",
        connection_string=PG_CONN,
    )

    print("✅ Ingesta terminada en 'manual_e5_multi'")

if __name__ == "__main__":
    main()
