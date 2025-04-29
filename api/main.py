# api/main.py
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.pgvector import PGVector
from langchain_deepseek import ChatDeepSeek

# Reranker

from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers import ContextualCompressionRetriever

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()

PG_CONN = os.environ["PG_CONN"]

# Embeddings: BGE-base, igual que en ingest.py
emb = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={"device": "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu"},
)

# Vector store y retriever base (k = 10 para rerank)
vectorstore = PGVector(
    embedding_function=emb,
    collection_name="manual_bge_base",
    connection_string=PG_CONN,
)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 30})

cross_enc = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
reranker = CrossEncoderReranker(model=cross_enc, top_n=4)

retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=base_retriever,
)

# LLM DeepSeek
llm = ChatDeepSeek(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    model="deepseek-chat",
    temperature=0,
)

# Prompt con guardrail
prompt_tmpl = """
Eres un asistente especializado en consultar el manual proporcionado.
Responde **solo** con información del CONTEXTO.
Si no encuentras la respuesta, di exactamente:
"La información solicitada no se encuentra en este manual."

CONTEXTO:
{context}

PREGUNTA:
{question}

RESPUESTA:
"""
prompt = PromptTemplate(template=prompt_tmpl, input_variables=["context", "question"])

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True,
)

# -------- FastAPI ----------
app = FastAPI(title="RAG-as-a-Service")

class Question(BaseModel):
    query: str
    k: int | None = None  # opcionalmente sobre-escribe k

@app.post("/chat")
def chat(q: Question):
    try:
        # pasa k si llega
        kwargs = {"k": q.k} if q.k else {}
        result = qa_chain({"query": q.query}, kwargs)
        return {
            "answer": result["result"],
            "sources": [
                {
                    "file": d.metadata.get("source"),
                    "snippet": d.page_content[:200] + "…",
                }
                for d in result["source_documents"]
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
