# api/main.py
import os
import time
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from functools import lru_cache
import json

from fastapi import FastAPI, HTTPException, Depends, Request, status, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ConfigDict, validator, root_validator
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

from transformers import pipeline
from langchain_community.vectorstores.pgvector import PGVector
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers import ContextualCompressionRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_deepseek import ChatDeepSeek
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError
from fastapi.middleware.cors import CORSMiddleware
from langchain.schema import Document
from datetime import datetime, timedelta
from fastapi import Depends, HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware
import time
from collections import defaultdict
from langdetect import detect, LangDetectException
import psycopg2
import traceback
import hashlib
import sys
import multiprocessing
import psutil
import torch


# MODIFICADO: Importar dependencias adicionales para health check y rendimiento
from starlette.status import HTTP_503_SERVICE_UNAVAILABLE
import asyncio
from contextlib import asynccontextmanager
from pydantic import model_validator
# Import section at top of file (add after other imports)
import nest_asyncio
nest_asyncio.apply()  # Apply nest_asyncio early to allow nested event loops

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ───── Configuración ─────
class Settings(BaseSettings):
    pg_conn: str
    deepseek_api_key: str
    hf_token: Optional[str] = None
    reranker_model: str = "BAAI/bge-reranker-base"
    embedding_model: str = "intfloat/multilingual-e5-base"
    llm_model: str = "deepseek-chat"
    collection_name: str = "manual_e5_multi"
    initial_retrieval_k: int = 20
    final_docs_count: int = 4
    api_key: Optional[str] = None
    cache_ttl_hours: int = 6
    environment: str = "production"  # MODIFICADO: Añadido para distinguir entornos
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])  # MODIFICADO: Configuración de CORS
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# MODIFICADO: Gestión de lifespan para cierre limpio de recursos
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Inicialización
    logger.info("Iniciando aplicación API RAG")
    load_dotenv()
    
    # Entregar control a la aplicación
    yield
    
    # Limpieza al cierre
    logger.info("Cerrando recursos y conexiones")
    # Aquí se podrían cerrar pools de conexiones, etc.

# MODIFICADO: Mejor manejo de errores en la carga de configuración
def load_settings() -> Settings:
    """Carga la configuración con manejo de errores mejorado"""
    try:
        settings = Settings()
        return settings
    except Exception as e:
        logger.critical(f"Error crítico al cargar configuración: {e}")
        msg = f"Fallo en la configuración. Verifica las variables de entorno: {str(e)}"
        raise RuntimeError(msg)

try:
    settings = load_settings()
except Exception as e:
    logger.critical(f"No se pudo cargar la configuración. La aplicación no iniciará: {e}")
    raise

# MODIFICADO: Creación de la aplicación con lifespan
app = FastAPI(
    title="RAG API",
    description="API para consultas RAG sobre documentación técnica",
    version="1.0.0",
    lifespan=lifespan
)

# MODIFICADO: Agregar middleware CORS configurado desde settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ───── Middlewares y seguridad ─────
# MODIFICADO: Middleware para métricas y logging
class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Extraer información de la petición
        path = request.url.path
        method = request.method
        
        # Procesar la petición
        try:
            response = await call_next(request)
            status_code = response.status_code
            
            # Calcular tiempo de respuesta
            duration = time.time() - start_time
            
            # Logear métricas
            logger.info(
                f"Request: {method} {path} | Status: {status_code} | "
                f"Duration: {duration:.4f}s"
            )
            
            return response
        except Exception as e:
            # Logear excepciones no manejadas
            logger.error(f"Error no manejado: {method} {path} | Error: {str(e)}")
            duration = time.time() - start_time
            logger.info(f"Request failed: {method} {path} | Duration: {duration:.4f}s")
            
            # Devolver error 500
            return JSONResponse(
                status_code=500,
                content={"detail": "Error interno del servidor"}
            )

# Añadir middleware de métricas
app.add_middleware(MetricsMiddleware)

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

# MODIFICADO: Límite de tasa con memoria en caché
class RateLimiter:
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.request_history = defaultdict(list)
        
    async def check_rate_limit(self, request: Request):
        # Obtener cliente IP o API key como identificador
        client_id = request.headers.get("X-API-Key", request.client.host)
        
        # Obtener tiempo actual
        now = time.time()
        
        # Filtrar solicitudes recientes (último minuto)
        minute_ago = now - 60
        self.request_history[client_id] = [
            t for t in self.request_history[client_id] if t > minute_ago
        ]
        
        # Verificar límite
        if len(self.request_history[client_id]) >= self.requests_per_minute:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Límite de velocidad excedido. Intente nuevamente más tarde."
            )
            
        # Registrar esta solicitud
        self.request_history[client_id].append(now)
        return True

# Instanciar limitador de tasa
rate_limiter = RateLimiter(requests_per_minute=60)

# ───── Modelos de carga perezosa ─────
@lru_cache(maxsize=1)
def get_question_rewriter():
    """Inicializa el modelo de reescritura sólo cuando se necesita"""
    logger.info("Inicializando modelo de reescritura de consultas")
    try:
        return pipeline(
            "text2text-generation",
            model="google/flan-t5-small",
            device_map="cpu",
            trust_remote_code=False,
        )
    except Exception as e:
        logger.error(f"Error al inicializar modelo de reescritura: {e}")
        # Crear una función fallback que simplemente devuelve la entrada
        def fallback_rewriter(text):
            return [{"generated_text": text}]
        return fallback_rewriter

@lru_cache(maxsize=1)
def get_embeddings():
    """Inicializa y retorna el modelo de embeddings"""
    logger.info("Inicializando modelo de embeddings")
    try:
        return HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
        )
    except Exception as e:
        logger.error(f"Error al inicializar embeddings: {e}")
        raise RuntimeError(f"No se pudo inicializar el modelo de embeddings: {e}")

def get_llm():
    """
    Función de compatibilidad para mantener código existente.
    Proporciona acceso síncrono al LLM desde el caché.
    """
    import asyncio
    
    # Crear un nuevo evento loop si es necesario
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    # Ejecutar la función asíncrona y obtener el resultado
    try:
        return loop.run_until_complete(llm_cache.get_llm())
    except Exception as e:
        logger.error(f"Error al obtener LLM de forma síncrona: {e}")
        raise RuntimeError(f"No se pudo inicializar el modelo LLM: {e}")

# Replace lines 242-248 with this code:
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
    before=lambda retry_state: logger.warning(
        f"Reintentando conexión a PGVector (intento {retry_state.attempt_number}/3)..."
    )
)
@lru_cache(maxsize=1)
def get_vectorstore():
    """Inicializa y retorna el almacén vectorial con reintentos"""
    logger.info("Inicializando conexión a PGVector")
    try:
        return PGVector(
            embedding_function=get_embeddings(),
            collection_name=settings.collection_name,
            connection_string=settings.pg_conn,
        )
    except Exception as e:
        logger.error(f"Error al conectar con PGVector: {e}")
        raise RuntimeError(f"No se pudo inicializar la base de datos vectorial: {e}")

@lru_cache(maxsize=1)
def get_retriever():
    """Inicializa el retriever cuando se necesita"""
    logger.info("Inicializando retriever")
    try:
        vectorstore = get_vectorstore()
        
        base_retriever = vectorstore.as_retriever(
            search_kwargs={"k": settings.initial_retrieval_k}
        )
        
        cross_enc = HuggingFaceCrossEncoder(
            model_name=settings.reranker_model,
            model_kwargs={"device": "cpu"}
        )
        reranker = CrossEncoderReranker(
            model=cross_enc, 
            top_n=settings.final_docs_count
        )
        
        return ContextualCompressionRetriever(
            base_compressor=reranker,
            base_retriever=base_retriever,
        )
    except Exception as e:
        logger.error(f"Error al inicializar retriever: {e}", exc_info=True)
        raise RuntimeError(f"No se pudo inicializar el retriever: {e}")

# MODIFICADO: Sistema de prompts mejor estructurado
def get_qa_prompt():
    """Devuelve el prompt para el sistema QA"""
    return PromptTemplate(
        template="""
Eres un asistente especializado en responder preguntas basándote en la información proporcionada.

CONTEXTO:
{context}

PREGUNTA:
{question}

Instrucciones:
1. Responde de manera completa y detallada usando SOLO la información del CONTEXTO.
2. Si el CONTEXTO contiene toda la información necesaria, proporciona una respuesta completa.
3. Si el CONTEXTO contiene información parcial, proporciona la parte que puedas responder e indica qué aspectos no están cubiertos.
4. Si el CONTEXTO no contiene información relevante, responde: "La información solicitada no se encuentra en este manual."
5. No inventes información ni uses conocimiento externo.
6. Formatea la respuesta de manera legible usando Markdown cuando sea apropiado.

RESPUESTA:
""",
        input_variables=["context", "question"],
    )

@lru_cache(maxsize=1)
def get_qa_chain():
    """Inicializa la cadena QA con el prompt mejorado"""
    return RetrievalQA.from_chain_type(
        llm=get_llm(),
        retriever=get_retriever(),
        chain_type="stuff",
        chain_type_kwargs={"prompt": get_qa_prompt()},
        return_source_documents=True,
    )

@lru_cache(maxsize=1)
def get_document_language_stats():
    """
    Analiza y devuelve estadísticas sobre los idiomas de los documentos en la base de datos.
    Cachea los resultados para evitar consultas repetidas.
    """
    logger.info("Analizando estadísticas de idiomas en la base de datos...")
    try:
        # MODIFICADO: Usar conexión directa a Postgres en lugar de PGVector
        conn = psycopg2.connect(settings.pg_conn)
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT metadata->>'language' as lang, COUNT(*) as count
                FROM langchain_pg_embedding
                WHERE collection_name = %s AND metadata->>'language' IS NOT NULL
                GROUP BY metadata->>'language'
                ORDER BY count DESC
            """, (settings.collection_name,))
            
            results = cursor.fetchall()
        conn.close()
        
        # Organizar resultados
        lang_stats = {}
        for lang, count in results:
            lang_stats[lang] = count
            
        logger.info(f"Estadísticas de idiomas: {lang_stats}")
        return lang_stats
        
    except Exception as e:
        logger.error(f"Error al obtener estadísticas de idiomas: {e}")
        return {}  # En caso de error, devolver diccionario vacío

# MODIFICADO: Función más robusta para fusionar documentos
def merge_document_info(documents: List[Document]) -> List[Document]:
    """
    Fusiona la información de múltiples documentos para proporcionar un contexto
    más coherente al LLM. Agrupa chunks del mismo documento.
    """
    if not documents:
        return []
        
    # MODIFICADO: Validación defensiva de entrada
    valid_docs = [doc for doc in documents if isinstance(doc, Document)]
    if len(valid_docs) != len(documents):
        logger.warning(f"Se descartaron {len(documents) - len(valid_docs)} documentos inválidos")
    
    # Agrupar por fuente/documento original
    docs_by_source = {}
    for doc in valid_docs:
        source = doc.metadata.get("source", "unknown")
        if source not in docs_by_source:
            docs_by_source[source] = []
        docs_by_source[source].append(doc)
    
    # Para cada fuente, ordenar chunks y fusionar si son consecutivos
    merged_docs = []
    for source, chunks in docs_by_source.items():
        # Intentar ordenar por número de página si existe
        try:
            chunks.sort(key=lambda x: x.metadata.get("page", 0))
            # MODIFICADO: Ordenar también por chunk_id para asegurar orden correcto
            chunks.sort(key=lambda x: x.metadata.get("chunk_id", 0))
        except Exception as e:
            logger.warning(f"Error al ordenar chunks para {source}: {e}")
        
        if not chunks:
            continue
            
        current_content = chunks[0].page_content
        current_meta = chunks[0].metadata.copy()
        
        for i in range(1, len(chunks)):
            # Si los chunks parecen ser consecutivos, fusionar
            if is_consecutive(chunks[i-1], chunks[i]):
                current_content += "\n\n" + chunks[i].page_content
                # Actualizar metadatos como rango de páginas
                if "page" in chunks[i].metadata and "page" in current_meta:
                    current_meta["page_range"] = f"{current_meta.get('page', 0)}-{chunks[i].metadata['page']}"
            else:
                # Si no son consecutivos, guardar el actual y empezar uno nuevo
                merged_docs.append(Document(page_content=current_content, metadata=current_meta))
                current_content = chunks[i].page_content
                current_meta = chunks[i].metadata.copy()
        
        # No olvidar el último documento
        merged_docs.append(Document(page_content=current_content, metadata=current_meta))
    
    return merged_docs

def is_consecutive(doc1: Optional[Document], doc2: Optional[Document]) -> bool:
    """Determina si dos documentos son consecutivos basándose en metadatos"""
    if not doc1 or not doc2:  # Validación defensiva
        return False
        
    # Si son del mismo archivo y tienen información de página
    if doc1.metadata.get("source") == doc2.metadata.get("source"):
        # Si tienen páginas especificadas, verificar si son consecutivas
        if "page" in doc1.metadata and "page" in doc2.metadata:
            try:
                return doc2.metadata["page"] - doc1.metadata["page"] <= 1
            except (TypeError, ValueError):
                pass  # Seguir con otras comprobaciones si falla la conversión numérica
        
        # Si tienen posición en el texto (chunk_id), verificar si son consecutivos
        if "chunk_id" in doc1.metadata and "chunk_id" in doc2.metadata:
            try:
                return doc2.metadata["chunk_id"] - doc1.metadata["chunk_id"] == 1
            except (TypeError, ValueError):
                pass
            
    # Si no hay metadatos útiles, usar el enfoque basado en contenido solo como fallback
    return False

# MODIFICADO: Sistema de caché mejorado con expiración y manejo avanzado
class LLMCache:
    """Gestiona el caché del LLM con expiración controlada"""
    
    def __init__(self, ttl_hours: int = 6):
        self._cache = {}
        self._timestamps = {}
        self._ttl = timedelta(hours=ttl_hours)
        self._lock = asyncio.Lock()
    
    async def get_llm(self):
        """Obtiene el LLM del caché o crea uno nuevo si expiró"""
        cache_key = "llm"
        now = datetime.now()
    
        async with self._lock:
            # Verificar caché
            if (cache_key in self._timestamps and 
                now - self._timestamps[cache_key] <= self._ttl and
                cache_key in self._cache):
                return self._cache[cache_key]
        
        # Crear nuevo LLM
            try:
            # CORREGIDO: Configurar la API key correctamente
            # La API key debe configurarse según la documentación de langchain_deepseek
                llm = ChatDeepSeek(
                    model=settings.llm_model,
                    api_key=settings.deepseek_api_key  # Usar api_key directamente en lugar de model_kwargs
                )
                self._cache[cache_key] = llm
                self._timestamps[cache_key] = now
                return llm
            except Exception as e:
                logger.error(f"Error al inicializar LLM: {e}")
                raise RuntimeError(f"No se pudo inicializar el modelo LLM: {e}")
    
    def invalidate(self):
        """Invalida explícitamente el caché"""
        self._cache.clear()
        self._timestamps.clear()

# Instanciar el caché de LLM
llm_cache = LLMCache(ttl_hours=settings.cache_ttl_hours)

# MODIFICADO: Función verificadora de respuestas mejorada con timeout
async def verify_answer_quality(question: str, answer: str, context_docs: List[Document]) -> Dict[str, Any]:
    """
    Verifica la calidad de la respuesta producida por el LLM,
    identificando si es completa o parcial respecto al contexto.
    Incluye manejo de timeouts y mecanismos más estrictos contra alucinaciones.
    """
    try:
        # Obtener LLM con caché
        llm = await llm_cache.get_llm()
        
        # Extraer el contexto relevante
        context = "\n\n".join([doc.page_content for doc in context_docs])
        
        # 1. Primero, verificar si la respuesta contiene información que NO está en el contexto (alucinación)
        hallucination_check_prompt = f"""
Analiza si la siguiente respuesta contiene información que NO está presente en el contexto proporcionado.

PREGUNTA: {question}

RESPUESTA GENERADA: {answer}

CONTEXTO DISPONIBLE: {context}

INSTRUCCIONES:
1. Identifica CUALQUIER fragmento de información en la respuesta que no esté explícitamente mencionado en el contexto.
2. Para cada fragmento identificado, indica por qué consideras que es una alucinación.
3. Finalmente, determina:
   a) Si la respuesta contiene SOLO información presente en el contexto (respuesta = "SIN ALUCINACIONES")
   b) Si la respuesta contiene información no presente en el contexto (respuesta = "CONTIENE ALUCINACIONES")
"""

        # Ejecutar verificación con timeout
        try:
            hallucination_check = await asyncio.wait_for(
                asyncio.to_thread(lambda: llm.invoke(hallucination_check_prompt).content),
                timeout=10.0  # 10 segundos de timeout
            )
        except asyncio.TimeoutError:
            logger.warning("Timeout en verificación de alucinaciones")
            hallucination_check = "Verificación no concluyente por timeout"
        
        # Analizar resultado
        has_hallucination = "CONTIENE ALUCINACIONES" in hallucination_check or "alucinación" in hallucination_check.lower()
        
        # 2. Verificar si la respuesta es completa con respecto a la pregunta
        completeness_check_prompt = f"""
Evalúa si la siguiente respuesta aborda COMPLETAMENTE la pregunta planteada, considerando el contexto disponible.

PREGUNTA: {question}

RESPUESTA: {answer}

CONTEXTO DISPONIBLE: {context}

INSTRUCCIONES:
1. Determina si la respuesta aborda todos los aspectos de la pregunta que pueden ser respondidos con el contexto disponible.
2. Responde "COMPLETA" si la respuesta cubre todos los aspectos que pueden ser respondidos según el contexto.
3. Responde "PARCIAL" si la respuesta sólo cubre algunos aspectos de la pregunta que pueden responderse con el contexto.
4. Responde "INSUFICIENTE" si el contexto no contiene la información necesaria para responder la pregunta.
"""

        # Ejecutar verificación con timeout
        try:
            completeness_check = await asyncio.wait_for(
                asyncio.to_thread(lambda: llm.invoke(completeness_check_prompt).content),
                timeout=10.0
            )
        except asyncio.TimeoutError:
            logger.warning("Timeout en verificación de completitud")
            completeness_check = "PARCIAL (verificación no concluyente por timeout)"
            
        # Analizar completitud
        is_complete = "COMPLETA" in completeness_check
        is_partial = "PARCIAL" in completeness_check
        is_insufficient = "INSUFICIENTE" in completeness_check
        
        # Determinar calidad general
        quality_assessment = "alta"
        if has_hallucination:
            quality_assessment = "baja"
        elif is_partial:
            quality_assessment = "media"
        elif is_insufficient:
            quality_assessment = "baja"
            
        # MODIFICADO: Mejorar estructura de la información de verificación
        verification_info = {
            "quality": quality_assessment,
            "has_hallucination": has_hallucination,
            "completeness": "completa" if is_complete else "parcial" if is_partial else "insuficiente",
            "hallucination_details": hallucination_check if has_hallucination else None,
            "completeness_details": completeness_check
        }
        
        return verification_info
        
    except Exception as e:
        logger.error(f"Error en verificación de respuesta: {e}")
        return {
            "quality": "no_verificada",
            "error": str(e),
            "error_type": type(e).__name__
        }

# ───── Modelos de datos ─────
class QueryRequest(BaseModel):
    query: str
    language: Optional[str] = None
    max_sources: Optional[int] = Field(None, ge=1, le=10)
    detailed_feedback: bool = False

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "¿Cómo funciona la recuperación de documentos en este sistema?",
                "language": "es",
                "max_sources": 4,
                "detailed_feedback": True
            }
        }
    )
    
    # MODIFICADO: Validador para sanitizar la entrada
    @validator('query')
    def sanitize_query(cls, v):
        if not v or not v.strip():
            raise ValueError("La consulta no puede estar vacía")
        # Limitar longitud de la consulta
        if len(v) > 1000:
            raise ValueError("La consulta excede el límite de 1000 caracteres")
        return v.strip()
    

    @root_validator(pre=True)
    def infer_language(cls, values):
        query = values.get('query')
        language = values.get('language')

        if not language and query:
            try:
                detected_lang = detect(query)
                values['language'] = detected_lang
            except LangDetectException:
                values['language'] = 'en'

        return values


class SourceInfo(BaseModel):
    source: str
    title: Optional[str] = None
    page: Optional[int] = None
    page_range: Optional[str] = None
    relevance_score: Optional[float] = None
    content_preview: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceInfo]
    execution_time: float
    verification: Optional[Dict[str, Any]] = None
    feedback_url: Optional[str] = None

# ───── Endpoints ─────
@app.get("/health")
async def health_check():
    """Endpoint para verificar salud del servicio"""
    try:
        # Verificar conexión a la base de datos
        vectorstore = get_vectorstore()
        
        # Verificar disponibilidad del LLM
        llm = await llm_cache.get_llm()
        
        # Si llegamos aquí, todo está bien
        return {
            "status": "ok",
            "database": "connected",
            "llm": "available",
            "timestamp": datetime.now().isoformat(),
            "environment": settings.environment
        }
    except Exception as e:
        logger.error(f"Health check fallido: {e}")
        return JSONResponse(
            status_code=HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

@app.get("/stats")
async def get_stats(request: Request):
    """Devuelve estadísticas básicas del sistema"""
    try:
        # Verificar autenticación
        await verify_api_key_if_configured(request)
        
        # Obtener estadísticas de idiomas
        lang_stats = get_document_language_stats()
        
        # MODIFICADO: Obtener también recuento total de documentos
        conn = psycopg2.connect(settings.pg_conn)
        total_documents = 0
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT COUNT(*) FROM langchain_pg_embedding 
                WHERE collection_name = %s
            """, (settings.collection_name,))
            total_documents = cursor.fetchone()[0]
        conn.close()
        
        return {
            "total_documents": total_documents,
            "language_distribution": lang_stats,
            "collection_name": settings.collection_name,
            "embedding_model": settings.embedding_model,
            "reranker_model": settings.reranker_model
        }
    except HTTPException:
        # Reenviar excepciones HTTP (como 401)
        raise
    except Exception as e:
        logger.error(f"Error al obtener estadísticas: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error al obtener estadísticas: {str(e)}"
        )

@app.post("/query", response_model=QueryResponse)
async def query_documents(
    request_data: QueryRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    api_auth: bool = Depends(verify_api_key_if_configured),
    rate_limit: bool = Depends(rate_limiter.check_rate_limit)
):
    """Procesa una consulta y devuelve la respuesta del
    modelo de lenguaje y los documentos relevantes"""
    start_time = time.time()
    
    # MODIFICADO: Validar que la consulta no esté vacía antes de procesar
    if not request_data.query or not request_data.query.strip():
        raise HTTPException(status_code=400, detail="La consulta no puede estar vacía")
    
    try:
        # MODIFICADO: Registro estructurado y seguro de la consulta (sin datos sensibles)
        query_log = {
            "query_id": f"q-{int(time.time())}",
            "timestamp": datetime.now().isoformat(),
            "query_length": len(request_data.query),
            "detected_language": request_data.language,
            "client_ip": request.client.host if request.client else "unknown"
        }
        logger.info(f"Nueva consulta: {json.dumps(query_log)}")
        
        # MODIFICADO: Reescritura opcional de la consulta para mejorar recuperación
        query = request_data.query
        try:
            # Obtener modelo de reescritura
            rewriter = get_question_rewriter()
            
            # Generar prompt para reescritura
            rewrite_prompt = f"Reescribe esta pregunta para optimizar la recuperación de información relevante: {query}"
            
            # Ejecutar reescritura con timeout
            rewritten_result = await asyncio.wait_for(
                asyncio.to_thread(lambda: rewriter(rewrite_prompt)[0]["generated_text"]),
                timeout=3.0  # Timeout de 3 segundos para no retrasar mucho
            )
            
            # Solo usar si la reescritura no está vacía y es diferente
            if rewritten_result and rewritten_result.strip() and rewritten_result != query:
                logger.debug(f"Consulta reescrita: {rewritten_result}")
                # Usar la consulta original y la reescrita
                query = f"{query} {rewritten_result}"
        except (asyncio.TimeoutError, Exception) as e:
            # Si hay error en reescritura, seguir con consulta original
            logger.warning(f"Error en reescritura de consulta: {str(e)}")
        
        # Obtener documentos relevantes
        try:
            # MODIFICADO: Optimización de la cadena QA para mejor rendimiento
            chain = get_qa_chain()
            
            # Ejecutar la consulta con manejo de errores
            result = chain({"query": query})
            
            # Extraer respuesta y documentos fuente
            answer = result.get("result", "")
            raw_source_docs = result.get("source_documents", [])
            
            # MODIFICADO: Mejorar procesamiento de fuentes con merge_document_info
            source_docs = merge_document_info(raw_source_docs)
            
            # Limitar número de fuentes si se especifica
            max_sources = request_data.max_sources or settings.final_docs_count
            source_docs = source_docs[:max_sources]
            
            # MODIFICADO: Formatear información de fuentes con vistas previas
            sources_info = []
            for i, doc in enumerate(source_docs):
                # Extraer metadatos relevantes con manejo defensivo
                metadata = doc.metadata or {}
                
                # Crear vista previa del contenido (primeros 150 caracteres)
                content_preview = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
                
                # Añadir información de fuente formateada
                sources_info.append(SourceInfo(
                    source=metadata.get("source", f"source-{i}"),
                    title=metadata.get("title"),
                    page=metadata.get("page"),
                    page_range=metadata.get("page_range"),
                    relevance_score=metadata.get("score"),
                    content_preview=content_preview if request_data.detailed_feedback else None
                ))
            
            # MODIFICADO: Verificación opcional de calidad de respuesta
            verification_info = None
            if request_data.detailed_feedback:
                verification_info = await verify_answer_quality(
                    question=request_data.query,
                    answer=answer,
                    context_docs=source_docs
                )
            
            # Calcular tiempo de ejecución
            execution_time = time.time() - start_time
            
            # MODIFICADO: Añadir enlace opcional de feedback si está configurado
            feedback_url = None
            if os.environ.get("FEEDBACK_URL"):
                feedback_id = hashlib.md5(f"{request_data.query}-{int(time.time())}".encode()).hexdigest()[:10]
                feedback_url = f"{os.environ.get('FEEDBACK_URL')}?id={feedback_id}"
            
            # MODIFICADO: Registro de telemetría en background para no bloquear respuesta
            background_tasks.add_task(
                log_query_metrics,
                query=request_data.query,
                num_sources=len(sources_info),
                execution_time=execution_time,
                language=request_data.language,
                quality=verification_info.get("quality", "no_verificada") if verification_info else "no_verificada"
            )
            
            # Devolver respuesta formateada
            return QueryResponse(
                answer=answer,
                sources=sources_info,
                execution_time=execution_time,
                verification=verification_info,
                feedback_url=feedback_url
            )
            
        except Exception as e:
            # Capturar errores específicos de la cadena
            logger.error(f"Error en procesamiento de consulta: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Error al procesar la consulta: {str(e)}"
            )
    
    except HTTPException:
        # Re-lanzar excepciones HTTP
        raise
    except Exception as e:
        # Capturar cualquier otro error
        logger.error(f"Error general en endpoint /query: {e}", exc_info=True)
        traceback_str = traceback.format_exc()
        logger.debug(f"Traceback: {traceback_str}")
        
        raise HTTPException(
            status_code=500,
            detail="Error interno del servidor al procesar la consulta."
        )

# MODIFICADO: Función para registro de métricas sin bloquear respuesta
async def log_query_metrics(
    query: str,
    num_sources: int,
    execution_time: float,
    language: str,
    quality: str
):
    """Registra métricas de la consulta de forma asíncrona"""
    try:
        # Aquí se podría enviar datos a un sistema de análisis
        # Como Prometheus, InfluxDB, etc.
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "query_length": len(query),
            "num_sources": num_sources,
            "execution_time": execution_time,
            "language": language,
            "quality": quality
        }
        
        logger.info(f"Metrics: {json.dumps(metrics)}")
        
        # Si hay una base de datos para métricas configurada, guardar
        if os.environ.get("METRICS_DB"):
            # Aquí iría código para guardar en BD de métricas
            pass
            
    except Exception as e:
        logger.error(f"Error al registrar métricas: {e}")
        # No re-lanzar excepción para no afectar al flujo principal

# MODIFICADO: Endpoint para búsqueda directa en vectorstore
@app.post("/search")
async def search_documents(
    request: Request,
    query: str = None,
    k: int = 5,
    api_auth: bool = Depends(verify_api_key_if_configured),
    rate_limit: bool = Depends(rate_limiter.check_rate_limit)
):
    """Endpoint para búsqueda directa sin procesamiento LLM"""
    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="La consulta no puede estar vacía")
    
    try:
        # Obtener vectorstore
        vectorstore = get_vectorstore()
        
        # Realizar búsqueda directa
        results = vectorstore.similarity_search_with_score(query, k=k)
        
        # Formatear resultados
        formatted_results = []
        for doc, score in results:
            # Extraer metadatos relevantes
            metadata = doc.metadata or {}
            
            # Normalizar score (0-1 donde 1 es mejor)
            normalized_score = float(score) if isinstance(score, (int, float)) else 0.0
            
            # Crear vista previa del contenido
            content_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            
            # Añadir resultado formateado
            formatted_results.append({
                "content": content_preview,
                "source": metadata.get("source", "unknown"),
                "page": metadata.get("page"),
                "score": normalized_score,
                "language": metadata.get("language", "unknown")
            })
        
        return {
            "results": formatted_results,
            "count": len(formatted_results),
            "query": query
        }
        
    except Exception as e:
        logger.error(f"Error en búsqueda directa: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error al realizar la búsqueda: {str(e)}"
        )

# MODIFICADO: Endpoint para mantenimiento de caché
@app.post("/admin/cache/invalidate", status_code=204)
async def invalidate_cache(
    request: Request,
    api_auth: bool = Depends(verify_api_key_if_configured)
):
    """Invalida los cachés de la aplicación"""
    # Verificar que haya una API key configurada para este endpoint sensible
    if not settings.api_key:
        raise HTTPException(
            status_code=403,
            detail="Este endpoint requiere autenticación con API key"
        )
    
    try:
        # Invalidar caché LRU
        get_embeddings.cache_clear()
        get_vectorstore.cache_clear()
        get_retriever.cache_clear()
        get_qa_chain.cache_clear()
        get_document_language_stats.cache_clear()
        
        # Invalidar caché de LLM
        llm_cache.invalidate()
        
        logger.info("Cachés invalidados por solicitud administrativa")
        return None
        
    except Exception as e:
        logger.error(f"Error al invalidar cachés: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error al invalidar cachés: {str(e)}"
        )

# MODIFICADO: Endpoint para diagnóstico de configuración (solo en dev/staging)
@app.get("/admin/diagnostics")
async def get_diagnostics(
    request: Request,
    api_auth: bool = Depends(verify_api_key_if_configured)
):
    """Devuelve información de diagnóstico para solución de problemas"""
    # Verificar que no estamos en producción
    if settings.environment.lower() == "production":
        raise HTTPException(
            status_code=403,
            detail="Este endpoint no está disponible en producción"
        )
    
    # Verificar que haya una API key configurada para este endpoint sensible
    if not settings.api_key:
        raise HTTPException(
            status_code=403,
            detail="Este endpoint requiere autenticación con API key"
        )
    
    try:
        # Recopilar información de diagnóstico
        diagnostics = {
            "settings": {
                "embedding_model": settings.embedding_model,
                "reranker_model": settings.reranker_model,
                "collection_name": settings.collection_name,
                "initial_retrieval_k": settings.initial_retrieval_k,
                "final_docs_count": settings.final_docs_count,
                "environment": settings.environment,
                "cache_ttl_hours": settings.cache_ttl_hours
            },
            "runtime": {
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "cpu_count": multiprocessing.cpu_count(),
                "memory_usage_mb": psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
            },
            "models_info": {
                "embedding_model_loaded": get_embeddings.cache_info().hits > 0,
                "vectorstore_connected": get_vectorstore.cache_info().hits > 0,
                "retriever_initialized": get_retriever.cache_info().hits > 0
            }
        }
        
        # Verificar conectividad a BD
        try:
            conn = psycopg2.connect(settings.pg_conn)
            diagnostics["database"] = {
                "connection": "ok",
                "backend_pid": conn.get_backend_pid()
            }
            conn.close()
        except Exception as db_error:
            diagnostics["database"] = {
                "connection": "error",
                "error": str(db_error)
            }
        
        return diagnostics
        
    except Exception as e:
        logger.error(f"Error al generar diagnóstico: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error al generar información de diagnóstico: {str(e)}"
        )

# MODIFICADO: Endpoint para verificar problemas comunes en configuración
@app.get("/admin/check")
async def check_configuration(
    request: Request,
    api_auth: bool = Depends(verify_api_key_if_configured)
):
    """Verifica problemas comunes en la configuración"""
    # Verificar que haya una API key configurada para este endpoint sensible
    if not settings.api_key:
        raise HTTPException(
            status_code=403,
            detail="Este endpoint requiere autenticación con API key"
        )
    
    issues = []
    warnings = []
    
    # Verificar variables de entorno críticas
    if not settings.pg_conn:
        issues.append("Variable PG_CONN no configurada")
    
    if not settings.deepseek_api_key:
        issues.append("Variable DEEPSEEK_API_KEY no configurada")
    
    # Verificar conexión a BD
    try:
        conn = psycopg2.connect(settings.pg_conn)
        with conn.cursor() as cursor:
            # Verificar si la colección existe
            cursor.execute("""
                SELECT COUNT(*) FROM langchain_pg_embedding 
                WHERE collection_name = %s
            """, (settings.collection_name,))
            count = cursor.fetchone()[0]
            if count == 0:
                warnings.append(f"La colección '{settings.collection_name}' está vacía")
        conn.close()
    except Exception as e:
        issues.append(f"Error de conexión a base de datos: {str(e)}")
    
    # Verificar modelos
    try:
        embeddings = get_embeddings()
    except Exception as e:
        issues.append(f"Error al inicializar modelo de embeddings: {str(e)}")
    
    # Verificar LLM
    try:
        llm = await llm_cache.get_llm()
    except Exception as e:
        issues.append(f"Error al inicializar LLM: {str(e)}")
    
    # Devolver resultados
    return {
        "status": "error" if issues else "warning" if warnings else "ok",
        "issues": issues,
        "warnings": warnings
    }


# MODIFICADO: Endpoint para obtener documentos por IDs
@app.get("/documents/{doc_id}")
async def get_document_by_id(
    doc_id: str,
    request: Request,
    api_auth: bool = Depends(verify_api_key_if_configured)
):
    """Obtiene un documento específico por su ID"""
    try:
        # Conectar a la base de datos directamente para mayor eficiencia
        conn = psycopg2.connect(settings.pg_conn)
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT id, document, metadata 
                FROM langchain_pg_embedding 
                WHERE collection_name = %s AND id = %s
            """, (settings.collection_name, doc_id))
            
            result = cursor.fetchone()
            
        conn.close()
        
        if not result:
            raise HTTPException(
                status_code=404,
                detail=f"Documento con ID {doc_id} no encontrado"
            )
            
        doc_id, document, metadata = result
        
        return {
            "id": doc_id,
            "content": document,
            "metadata": metadata
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error al obtener documento {doc_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error al recuperar documento: {str(e)}"
        )

# MODIFICADO: Punto de entrada para ejecución directa
if __name__ == "__main__":
    import uvicorn
    import sys
    
    # Obtener puerto desde argumentos o usar 8000 por defecto
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    
    # Mostrar información de configuración
    print(f"🚀 Iniciando API RAG en puerto {port}")
    print(f"📄 Colección: {settings.collection_name}")
    print(f"🧠 Modelo de embeddings: {settings.embedding_model}")
    print(f"🔄 Modelo de reranking: {settings.reranker_model}")
    print(f"🤖 LLM: {settings.llm_model}")
    print(f"🌍 Entorno: {settings.environment}")
    
    # Iniciar servidor
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        reload=settings.environment.lower() != "production"  # Habilitar recarga en desarrollo
    )