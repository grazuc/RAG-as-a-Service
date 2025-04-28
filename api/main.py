# api/main.py
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores.pgvector import PGVector
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_deepseek import ChatDeepSeek
from langchain.prompts import PromptTemplate


load_dotenv()  # carga PG_CONN y DEEPSEEK_API_KEY

# ---------- Configuración ----------
PG_CONN = os.environ["PG_CONN"]

# Embeddings deben ser IGUALES a los que usaste al indexar
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = PGVector(
    embedding_function=emb,
    collection_name="manual_demo",
    connection_string=PG_CONN,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

llm = ChatDeepSeek(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    model="deepseek-chat"   # o el ID de modelo que uses, p. ej. "deepseek-r1"
)


# ---------- Prompt con tu instrucción ----------
prompt_tmpl = """
Eres un asistente especializado en consultar el manual proporcionado.
Responde **exclusivamente** con la información encontrada en el CONTEXTO.
Si la respuesta no está en el manual, contesta exactamente:
"La información solicitada no se encuentra en este manual."

CONTEXTO:
{context}

PREGUNTA:
{question}

RESPUESTA (en español):
"""

prompt = PromptTemplate(
    template=prompt_tmpl,
    input_variables=["context", "question"],
)

# ---------- Cadena QA con prompt personalizado ----------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True,
)

# ---------- FastAPI ----------
app = FastAPI(title="RAG-as-a-Service")

class Question(BaseModel):
    query: str
    k: int | None = 4  # nº de trozos opcional

@app.post("/chat")
def chat(question: Question):
    try:
        result = qa_chain(
            {"query": question.query},
            {"k": question.k} if question.k else None,
        )
        answer = {
            "answer": result["result"],
            "sources": [
                {
                    "text": doc.page_content[:200] + "…",
                    "file": doc.metadata.get("source"),
                }
                for doc in result["source_documents"]
            ],
        }
        return answer
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
