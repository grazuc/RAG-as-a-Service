# api/main.py
import os
import time
import logging
from typing import List, Dict, Any, Optional
from functools import lru_cache

from fastapi import FastAPI, HTTPException, Depends, Request, status
from pydantic import BaseModel, Field, ConfigDict
from pydantic_settings import BaseSettings
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
from tenacity import retry, stop_after_attempt, wait_exponential

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ───── Configuración ─────
class Settings(BaseSettings):
    pg_conn: str
    deepseek_api_key: str
    hf_token: str  # Añadido para coincidir con las variables en .env
    reranker_model: str = "BAAI/bge-reranker-base"
    embedding_model: str = "intfloat/multilingual-e5-base"
    llm_model: str = "deepseek-chat"
    collection_name: str = "manual_e5_multi2"
    initial_retrieval_k: int = 20
    final_docs_count: int = 4
    # La api_key es opcional, si no se proporciona no se verificará
    api_key: Optional[str] = None
    
    class Config:
        env_file = ".env"

# Cargar configuración
load_dotenv()
try:
    settings = Settings()
except Exception as e:
    logger.error(f"Error al cargar configuración: {e}")
    raise RuntimeError("Fallo en la configuración. Verifica las variables de entorno.")

# ───── Middlewares y seguridad ─────
# Verificador de API key condicional
async def verify_api_key_if_configured(request: Request):
    """Verifica la API key solo si está configurada en settings"""
    if not settings.api_key:
        # Si no hay API key configurada, permitir el acceso
        return True
        
    api_key = request.headers.get("X-API-Key")
    if not api_key or api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key inválida"
        )
    return True

# ───── Modelos de carga perezosa ─────
@lru_cache(maxsize=1)
def get_question_rewriter():
    """Inicializa el modelo de reescritura sólo cuando se necesita"""
    logger.info("Inicializando modelo de reescritura de consultas")
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        device_map="cpu",
        trust_remote_code=False,
    )

@lru_cache(maxsize=1)
def get_retriever():
    """Inicializa el retriever cuando se necesita"""
    logger.info("Inicializando retriever")
    emb = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
    )
    
    try:
        vectorstore = PGVector(
            embedding_function=emb,
            collection_name=settings.collection_name,
            connection_string=settings.pg_conn,
        )
    except Exception as e:
        logger.error(f"Error al conectar con PGVector: {e}")
        raise RuntimeError(f"No se pudo inicializar la base de datos vectorial: {e}")
    
    base_retriever = vectorstore.as_retriever(
        search_kwargs={"k": settings.initial_retrieval_k}
    )
    
    # Se eliminó el parámetro 'device' que estaba causando el error
    cross_enc = HuggingFaceCrossEncoder(
        model_name=settings.reranker_model,
    )
    reranker = CrossEncoderReranker(
        model=cross_enc, 
        top_n=settings.final_docs_count
    )
    
    return ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=base_retriever,
    )

@lru_cache(maxsize=1)
def get_llm():
    """Inicializa el LLM cuando se necesita"""
    logger.info("Inicializando LLM")
    return ChatDeepSeek(
        api_key=settings.deepseek_api_key,
        model=settings.llm_model,
        temperature=0,
        request_timeout=30,
    )

@lru_cache(maxsize=1)
def get_qa_chain():
    """Ensambla e inicializa la cadena QA cuando se necesita"""
    logger.info("Inicializando cadena QA")
    prompt = PromptTemplate(
        template="""
Eres un asistente que solo responde con información del manual.
Si no existe, di "La información solicitada no se encuentra en este manual."
No inventes información ni uses conocimiento externo.

CONTEXTO:
{context}

PREGUNTA:
{question}

RESPUESTA:
""",
        input_variables=["context", "question"],
    )
    
    return RetrievalQA.from_chain_type(
        llm=get_llm(),
        retriever=get_retriever(),
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
)
def rewrite_query(question: str) -> str:
    """Reescribe la consulta para optimizarla, con reintentos"""
    # Verificar si debemos usar el rewriter o no
    # Para consultas cortas y directas, no reescribimos
    if len(question.split()) <= 8 and ("?" in question or any(w in question.lower() for w in ["como", "cómo", "que", "qué", "cuál", "cual", "donde", "dónde"])):
        logger.info(f"Consulta corta detectada, se omite reescritura: {question}")
        return question
    
    # Reescribe consultas más complejas
    try:
        question_rewriter = get_question_rewriter()
        prompt_text = (
            "Eres un experto en búsqueda semántica. Reformula esta consulta para hacerla más "
            "efectiva para recuperación semántica. Mantén todos los términos técnicos y "
            "específicos. No agregues información que no esté en la consulta original. "
            "Mantén la consulta en el mismo idioma.\n\n"
            f"Consulta original: {question}\n\n"
            "Consulta reformulada:"
        )
        
        out = question_rewriter(
            prompt_text,
            max_new_tokens=100,
            clean_up_tokenization_spaces=True
        )
        rewritten = out[0]["generated_text"].strip()
        
        # Validaciones de calidad para la consulta reescrita
        # 1. Verificar longitud mínima
        if len(rewritten.split()) < 2:
            logger.warning(f"Reescritura demasiado corta, usando original: {rewritten}")
            return question
            
        # 2. Verificar que no sea demasiado diferente (demasiadas palabras nuevas)
        original_words = set(w.lower() for w in question.split())
        rewritten_words = set(w.lower() for w in rewritten.split())
        new_words = rewritten_words - original_words
        
        # Si más del 50% son palabras nuevas, es sospechoso
        if len(new_words) > len(original_words) * 0.5:
            logger.warning(f"Demasiadas palabras nuevas en reescritura, usando original. Nuevas: {new_words}")
            return question
            
        # 3. Verificar que se mantengan las palabras clave/técnicas
        # Identificar posibles términos técnicos (palabras con mayúsculas o poco comunes)
        potential_technical_terms = [w for w in question.split() if w[0].isupper() or len(w) > 7]
        for term in potential_technical_terms:
            if term.lower() not in " ".join(rewritten.lower().split()):
                logger.warning(f"Término técnico '{term}' perdido en reescritura, usando original")
                return question
                
        logger.info(f"Consulta reescrita: '{question}' -> '{rewritten}'")
        return rewritten
        
    except Exception as e:
        logger.error(f"Error al reescribir consulta: {e}")
        # En caso de error, devolvemos la consulta original
        return question

# ───── FastAPI ─────
app = FastAPI(
    title="RAG-as-a-Service",
    description="API para consultas RAG sobre manuales",
    version="1.0.0",
)

# ───── Modelos de datos ─────
class Question(BaseModel):
    query: str = Field(..., min_length=3, max_length=1000)
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "¿Cómo configuro la red WiFi?",
            }
        }
    )

class Source(BaseModel):
    file: str
    snippet: str

class ChatResponse(BaseModel):
    original_query: str
    rewritten_query: str
    answer: str
    sources: List[Source]

# ───── Endpoints ─────
@app.get("/health")
async def health_check():
    """Endpoint para verificar el estado del servicio"""
    return {"status": "ok", "version": "1.0.0"}

@app.post("/chat", response_model=ChatResponse)
async def chat(q: Question, _: bool = Depends(verify_api_key_if_configured)):
    """Endpoint principal para consultas RAG"""
    start_time = time.time()
    logger.info(f"Procesando consulta: {q.query}")
    
    try:
        # 1. Reescribe la pregunta
        good_query = rewrite_query(q.query)
        logger.info(f"Consulta reescrita: {good_query}")
        
        # 2. Ejecuta el RAG con esa query reescrita
        result = get_qa_chain().invoke({"query": good_query})
        
        # 3. Si no hay resultados relevantes, intentar con la consulta original
        has_fallback = False
        if not result["source_documents"] or "no se encuentra en este manual" in result["result"].lower():
            logger.warning("Sin resultados relevantes con consulta reescrita, intentando con la original")
            # Solo si la consulta reescrita es diferente de la original
            if good_query != q.query:
                has_fallback = True
                result = get_qa_chain().invoke({"query": q.query})
        
        # 4. Formatear la respuesta
        response = ChatResponse(
            original_query=q.query,
            rewritten_query=good_query if not has_fallback else q.query,
            answer=result["result"],
            sources=[
                Source(
                    file=d.metadata.get("source", "desconocido"),
                    snippet=d.page_content[:200] + "…" if len(d.page_content) > 200 else d.page_content,
                )
                for d in result["source_documents"]
            ],
        )
        
        process_time = time.time() - start_time
        logger.info(f"Consulta procesada en {process_time:.2f}s")
        
        return response
        
    except Exception as e:
        logger.error(f"Error al procesar consulta: {str(e)}", exc_info=True)
        if "deepseek" in str(e).lower():
            raise HTTPException(
                status_code=503,
                detail="Error en el servicio de DeepSeek. Intenta más tarde."
            )
        elif "pgvector" in str(e).lower() or "database" in str(e).lower():
            raise HTTPException(
                status_code=503,
                detail="Error en la base de datos. Intenta más tarde."
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Error interno: {str(e)}"
            )

# Inicializar componentes al arranque (opcional)
@app.on_event("startup")
async def startup_event():
    logger.info("Iniciando servicio RAG")
    # Warm up de los modelos (opcional - comentar si prefieres lazy loading)
    # get_question_rewriter()
    # get_retriever()
    # get_llm()
    # get_qa_chain()