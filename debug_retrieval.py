# debug_retrieval.py
import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.pgvector import PGVector

# 1. Configuración
load_dotenv()
PG_CONN = os.environ["PG_CONN"]

# 2. Embedding model (igual que en main.py)
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={"device": "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu"},
)

# 3. Vectorstore
vectorstore = PGVector(
    embedding_function=embedding_model,
    collection_name="manual_bge_base",
    connection_string=PG_CONN,
)

# 4. Hacer una búsqueda manual
query = "pantalla táctil capacitiva"
results = vectorstore.similarity_search(query, k=10)

# 5. Mostrar resultados
for i, doc in enumerate(results):
    print(f"\n[Documento {i+1}]")
    print(doc.page_content[:500])  # mostramos los primeros 500 caracteres
    print("---")
