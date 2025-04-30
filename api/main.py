# api/main.py
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from transformers import pipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.pgvector import PGVector
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers import ContextualCompressionRetriever
from langchain_deepseek import ChatDeepSeek
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ───── Carga de env vars ─────
load_dotenv()
PG_CONN = os.environ["PG_CONN"]
DEEPSEEK_API_KEY = os.environ["DEEPSEEK_API_KEY"]

# Carga Flan-T5 Small (mantiene CPU)
question_rewriter = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
    device_map="cpu",
    trust_remote_code=False,
)

def rewrite_query(question: str) -> str:
    # Prompt genérico para todo tipo de docs
    prompt_text = (
        "Toma la siguiente pregunta de un usuario y reescríbela de forma breve y "
        "óptima para que sirva como consulta de búsqueda semántica:\n\n"
        f"Pregunta original: {question}\n\n"
        "Consulta optimizada:"
    )
    out = question_rewriter(
        prompt_text,
        max_new_tokens=60,
        clean_up_tokenization_spaces=True
    )
    rewritten = out[0]["generated_text"].strip()
    # Si sale muy corto o sin cambios, volvemos a la original
    if len(rewritten.split()) < 2:
        return question
    return rewritten


# ───── Embeddings + VectorStore ─────
emb = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-base",
    model_kwargs={"device": "cpu"},
)
vectorstore = PGVector(
    embedding_function=emb,
    collection_name="manual_e5_multi",
    connection_string=PG_CONN,
)

# ───── Retriever base + Reranker ─────
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 30})
cross_enc = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
reranker = CrossEncoderReranker(model=cross_enc, top_n=4)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=base_retriever,
)

# ───── LLM DeepSeek ─────
llm = ChatDeepSeek(
    api_key=DEEPSEEK_API_KEY,
    model="deepseek-chat",
    temperature=0,
)

# ───── Prompt con guardrail ─────
prompt = PromptTemplate(
    template="""
Eres un asistente que solo responde con información del manual.
Si no existe, di “La información solicitada no se encuentra en este manual.”

CONTEXTO:
{context}

PREGUNTA:
{question}

RESPUESTA:
""",
    input_variables=["context", "question"],
)

# ───── Cadena QA ─────
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=compression_retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True,
)

# ───── FastAPI ─────
app = FastAPI(title="RAG-as-a-Service")

class Question(BaseModel):
    query: str

@app.post("/chat")
def chat(q: Question):
    try:
        # 1. Reescribe la pregunta
        good_query = rewrite_query(q.query)

        # 2. Ejecuta el RAG con esa query
        result = qa_chain.invoke({"query": good_query})

        return {
            "original_query": q.query,
            "rewritten_query": good_query,
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

