"""
ingest.py
----------
Lee todos los PDFs que encuentre en la carpeta ./docs,
los trocea, genera embeddings con e5-mistral y
los guarda en la colección 'manual_demo' de PGVector.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.pgvector import PGVector
from langchain.embeddings import HuggingFaceEmbeddings

# 1) Carga variables de entorno (.env)
load_dotenv()
PG_CONN = os.environ["PG_CONN"]

# 2) Directorio con tus PDFs
PDF_DIR = Path("docs")

# 3) Embeddings: e5-mistral (open-source, muy bueno)
EMBED_MODEL = "intfloat/e5-mistral-7b-instruct"
embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    # Usa GPU si tenés; de lo contrario queda en CPU
    model_kwargs={"device": "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu"},
)

# 4) Configura cómo vamos a trocear el texto
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,      # aprox. medio folio
    chunk_overlap=100,   # solapamos un poco para no cortar ideas
    separators=["\n\n", "\n", " ", ""],
)

def ingest_pdf(pdf_path: Path):
    """Carga, trocea y devuelve una lista de Document objects."""
    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()  # lista de Document (uno por página)

    # Metadata: guardamos el nombre del archivo para trazar la fuente
    for d in docs:
        d.metadata["source"] = pdf_path.name

    # Troceamos la lista completa
    return splitter.split_documents(docs)

def main():
    print("🔍 Buscando PDFs en", PDF_DIR.resolve())
    pdf_files = list(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError("⚠️  No se encontraron PDFs en ./docs")

    all_chunks = []
    for pdf in pdf_files:
        print(f"📄 Ingestando {pdf.name} …")
        all_chunks.extend(ingest_pdf(pdf))

    print(f"✂️  Total de segmentos: {len(all_chunks)}")

    # 5) Conectamos a PGVector y guardamos
    store = PGVector.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        collection_name="manual_demo",        # puedes tener varias colecciones
        connection_string=PG_CONN,
    )

    print(f"✅ Ingestados {len(all_chunks)} chunks en 'manual_demo'")

if __name__ == "__main__":
    main()
