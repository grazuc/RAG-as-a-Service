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
from fastapi.middleware.cors import CORSMiddleware
from langchain.schema import Document



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
    collection_name: str = "manual_e5_multi7"
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
# En main.py, modifica el prompt de la cadena QA
def get_qa_chain():
    prompt = PromptTemplate(
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

@lru_cache(maxsize=1)
def get_document_language_stats():
    """
    Analiza y devuelve estadísticas sobre los idiomas de los documentos en la base de datos.
    Cachea los resultados para evitar consultas repetidas.
    """
    logger.info("Analizando estadísticas de idiomas en la base de datos...")
    try:
        # Obtener documentos
        vectorstore = PGVector(
            connection_string=settings.pg_conn,
            collection_name=settings.collection_name,
        )
        
        # Ejecutar consulta SQL directa para obtener estadísticas de idiomas
        conn = vectorstore.connection
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT metadata->>'language' as lang, COUNT(*) as count
                FROM langchain_pg_embedding
                WHERE collection_name = %s AND metadata->>'language' IS NOT NULL
                GROUP BY metadata->>'language'
                ORDER BY count DESC
            """, (settings.collection_name,))
            
            results = cursor.fetchall()
        
        # Organizar resultados
        lang_stats = {}
        for lang, count in results:
            lang_stats[lang] = count
            
        logger.info(f"Estadísticas de idiomas: {lang_stats}")
        return lang_stats
        
    except Exception as e:
        logger.error(f"Error al obtener estadísticas de idiomas: {e}")
        return {}  # En caso de error, devolver diccionario vacío

def merge_document_info(documents):
    """
    Fusiona la información de múltiples documentos para proporcionar un contexto
    más coherente al LLM. Agrupa chunks del mismo documento.
    """
    if not documents:
        return []
        
    # Agrupar por fuente/documento original
    docs_by_source = {}
    for doc in documents:
        source = doc.metadata.get("source", "unknown")
        if source not in docs_by_source:
            docs_by_source[source] = []
        docs_by_source[source].append(doc)
    
    # Para cada fuente, ordenar chunks y fusionar si son consecutivos
    merged_docs = []
    for source, chunks in docs_by_source.items():
        # Intentar ordenar por número de página si existe
        chunks.sort(key=lambda x: x.metadata.get("page", 0))
        
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

def is_consecutive(doc1, doc2):
    """Determina si dos documentos parecen ser consecutivos basado en sus metadatos"""
    # Si tienen páginas consecutivas
    if "page" in doc1.metadata and "page" in doc2.metadata:
        return doc2.metadata["page"] - doc1.metadata["page"] <= 1
    
    # Si son del mismo archivo y tienen contenido que parece continuar
    if doc1.metadata.get("source") == doc2.metadata.get("source"):
        # Simplificación: verificar si el final de uno conecta con el inicio del otro
        last_sentence = doc1.page_content.split(".")[-1].strip()
        first_sentence = doc2.page_content.split(".")[0].strip()
        
        # Si la última oración del primer doc está incompleta (sin punto)
        # y parece continuar en el segundo doc
        if (not last_sentence.endswith(".") and 
            len(last_sentence) > 5 and
            any(word in first_sentence for word in last_sentence.split()[-3:])):
            return True
    
    return False

# Modificación para get_llm() - Aumentar timeout
@lru_cache(maxsize=1)
def get_llm():
    """Inicializa el LLM cuando se necesita"""
    logger.info("Inicializando LLM")
    return ChatDeepSeek(
        api_key=settings.deepseek_api_key,
        model=settings.llm_model,
        temperature=0,
        request_timeout=60,  # Aumentado de 30 a 60 segundos
    )

# Modificación para verify_answer_quality() - Manejo de timeout
def verify_answer_quality(question, answer, context_docs):
    """
    Verifica la calidad de la respuesta producida por el LLM,
    identificando si es completa o parcial respecto al contexto.
    Incluye manejo de timeouts y mecanismos más estrictos contra alucinaciones.
    """
    try:
        llm = get_llm()  # Reutilizar el LLM configurado
        
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
   b) Si la respuesta contiene información que NO está en el contexto (respuesta = "CONTIENE ALUCINACIONES: [lista de alucinaciones]")

RESULTADO:
"""
        
        @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=3))
        def check_hallucinations():
            return llm.invoke(hallucination_check_prompt)
        
        try:
            hallucination_result = check_hallucinations()
            hallucination_text = hallucination_result.content.strip()
            
            # Si detectamos alucinaciones, generar una nueva respuesta
            if "CONTIENE ALUCINACIONES" in hallucination_text:
                logger.warning(f"Detectadas alucinaciones en respuesta: {hallucination_text}")
                
                # Prompt más estricto para generar una respuesta sin alucinaciones
                strict_answer_prompt = f"""
Genera una respuesta a la siguiente pregunta usando EXCLUSIVAMENTE la información proporcionada en el contexto.
NO incluyas NINGUNA información adicional que no esté explícitamente mencionada en el contexto.

PREGUNTA: {question}

CONTEXTO: {context}

INSTRUCCIONES IMPORTANTES:
1. Si el contexto NO contiene suficiente información para responder completamente, INDICA EXPLÍCITAMENTE qué partes no puedes responder.
2. Usa frases como "Según la información proporcionada..." o "El documento menciona que..."
3. NO elabores, NO infiera, NO extrapole más allá del contenido exacto del contexto.
4. Si necesitas decir "no sé" o "el contexto no lo menciona", HAZLO sin dudar.

RESPUESTA CORREGIDA:
"""
                
                @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=3))
                def get_strict_answer():
                    return llm.invoke(strict_answer_prompt)
                
                try:
                    strict_result = get_strict_answer()
                    return {
                        "quality": "corrected",
                        "original_answer": answer,
                        "improved_answer": strict_result.content.strip(),
                        "analysis": hallucination_text
                    }
                except Exception as e:
                    logger.error(f"Error al generar respuesta estricta: {e}")
                    # Si falla, marcamos la respuesta como no confiable
                    return {
                        "quality": "unreliable",
                        "original_answer": answer,
                        "improved_answer": "No se pudo generar una respuesta confiable basada en el contexto disponible.",
                        "analysis": hallucination_text
                    }
        except Exception as e:
            logger.warning(f"Error en verificación de alucinaciones: {e}")
        
        # 2. Verificar completitud de la respuesta (como estaba antes, pero con umbral más estricto)
        verification_prompt = f"""
Evalúa si la siguiente respuesta contesta completamente a la pregunta basándose en el contexto proporcionado.

PREGUNTA: {question}

RESPUESTA GENERADA: {answer}

CONTEXTO DISPONIBLE: {context}

Realiza este análisis paso a paso:
1. ¿La respuesta aborda todos los aspectos de la pregunta? (sí/no)
2. ¿La respuesta incluye toda la información relevante del contexto? (sí/no)
3. ¿Hay información importante en el contexto que no se incluyó en la respuesta? (sí/no)
4. Si la respuesta es parcial, ¿qué información específica falta?
5. IMPORTANTE: ¿La respuesta menciona algo que NO está en el contexto? (sí/no)

Finalmente, clasifica la respuesta como:
A. COMPLETA (responde totalmente la pregunta con toda la información relevante del contexto)
B. PARCIAL (responde solo algunos aspectos o falta información importante)
C. INSUFICIENTE (apenas responde o ignora aspectos críticos)
D. NO CONFIABLE (incluye información que no está en el contexto)

Responde con la clasificación final (A, B, C o D) seguida de una breve justificación.
"""
        
        @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=3))
        def get_verification_with_retry():
            return llm.invoke(verification_prompt)
        
        try:
            verification_result = get_verification_with_retry()
            result_text = verification_result.content.strip()
            
            # Analizar la clasificación
            if "D. NO CONFIABLE" in result_text or "D:" in result_text:
                quality = "unreliable"
            elif "A. COMPLETA" in result_text or "A:" in result_text:
                quality = "complete"
            elif "B. PARCIAL" in result_text or "B:" in result_text:
                quality = "partial"
            else:
                quality = "insufficient"
                
            # Si la respuesta es parcial, insuficiente o no confiable, mejorarla
            if quality != "complete":
                # Prompt más específico para mejorar respuestas
                improvement_prompt = f"""
La respuesta actual a la pregunta "{question}" es {quality}. 
Genera una nueva respuesta que:

1. Se base ÚNICAMENTE en la información del contexto proporcionado.
2. Sea clara en indicar qué partes de la pregunta NO pueden responderse con el contexto disponible.
3. NO incluya información que no está en el contexto.
4. Use un lenguaje que indique la fuente de la información ("según el manual", "el documento menciona", etc.)

CONTEXTO: {context}

RESPUESTA ANTERIOR (PROBLEMÁTICA): {answer}

RESPUESTA MEJORADA:
"""
                
                @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=3))
                def get_improvement_with_retry():
                    return llm.invoke(improvement_prompt)
                
                try:
                    improved_result = get_improvement_with_retry()
                    improved_answer = improved_result.content.strip()
                    
                    # Verificación final de la respuesta mejorada
                    if "unreliable" in quality:
                        # Añadir advertencia explícita
                        improved_answer = "NOTA: La respuesta original contenía información no verificable con el contexto disponible. Esta es una respuesta revisada:\n\n" + improved_answer
                    
                except Exception as e:
                    logger.warning(f"Error al generar respuesta mejorada: {e}")
                    improved_answer = None
                    
                    # Si falló la mejora y la respuesta es no confiable, descartarla completamente
                    if quality == "unreliable":
                        improved_answer = "Lo siento, no puedo proporcionar una respuesta confiable a esta pregunta con la información disponible en el manual."
                
                return {
                    "quality": quality,
                    "original_answer": answer,
                    "improved_answer": improved_answer,
                    "analysis": result_text
                }
            
            return {
                "quality": quality,
                "original_answer": answer,
                "improved_answer": None,
                "analysis": result_text
            }
        except Exception as e:
            logger.warning(f"Error en verificación con retry: {e}")
            # Si falla incluso con retry, ser conservador
            return {
                "quality": "unknown",
                "original_answer": answer,
                "improved_answer": "No se pudo verificar adecuadamente la respuesta. La información proporcionada podría ser incompleta o imprecisa.",
                "analysis": "No se pudo verificar la calidad debido a un timeout"
            }
        
    except Exception as e:
        logger.error(f"Error en verificación de respuesta: {e}")
        return {
            "quality": "error",
            "original_answer": answer,
            "improved_answer": "Se produjo un error al verificar esta respuesta. Por favor, considere la información como potencialmente incompleta.",
            "analysis": f"Error en verificación: {str(e)}"
        }

# Simplificar la función rewrite_query en main.py
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def rewrite_query(question: str) -> str:
    """Versión mejorada que maneja mejor consultas con signos de interrogación"""
    # Solo omitir reescritura para consultas muy simples y directas
    if len(question.split()) <= 3:  # Solo consultas extremadamente cortas
        logger.info(f"Consulta demasiado simple, se omite reescritura: {question}")
        return question
    
    try:
        llm = get_llm()
        prompt_text = (
            "Reformula esta consulta para mejorar la recuperación semántica. "
            "Elimina saludos, frases irrelevantes y ruido. Mantén todos los términos técnicos y palabras clave. "
            "Mantén el mismo idioma. Responde ÚNICAMENTE con la consulta reformulada.\n\n"
            f"Consulta original: {question}\n\n"
            "Consulta reformulada:"
        )
        
        response = llm.invoke(prompt_text)
        rewritten = response.content.strip()
        
        # Verificaciones de calidad
        if len(rewritten.split()) < 2 or len(rewritten) < len(question) * 0.3:
            return question
            
        return rewritten
    except Exception as e:
        logger.error(f"Error al reescribir consulta: {e}")
        return question

def detect_language(text: str) -> Optional[str]:
    """Detecta el idioma del texto, devuelve None si falla"""
    if not text or len(text.strip()) < 20:
        return None
    
    try:
        return detect(text)
    except Exception as e:
        logger.error(f"Error al detectar idioma: {e}")
        return None

# ───── FastAPI ─────
app = FastAPI(
    title="RAG-as-a-Service",
    description="API para consultas RAG sobre manuales",
    version="1.0.0",
)

# Cors
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # URL de tu frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
    """Endpoint principal mejorado para consultas RAG"""
    start_time = time.time()
    logger.info(f"Procesando consulta: {q.query}")
    
    try:
        # 1. Reescribe la pregunta (versión mejorada)
        good_query = rewrite_query(q.query)
        logger.info(f"Consulta reescrita: {good_query}")
        
        # 2. Recupera documentos relevantes
        retriever = get_retriever()
        docs = retriever.get_relevant_documents(good_query)
        
        if not docs:
            logger.warning("Sin resultados con consulta reescrita, intentando con la original")
            if good_query != q.query:
                docs = retriever.get_relevant_documents(q.query)
                good_query = q.query  # Actualizamos para la respuesta
        
        if not docs:
            logger.warning("Sin resultados relevantes para la consulta")
            return ChatResponse(
                original_query=q.query,
                rewritten_query=good_query,
                answer="No se encontró información relevante en el manual para esta consulta.",
                sources=[]
            )
        
        # 3. Fusionar información de chunks relacionados
        merged_docs = merge_document_info(docs)
        logger.info(f"Documentos fusionados: {len(merged_docs)} a partir de {len(docs)} originales")
        
        # 4. Generar respuesta con el LLM
        context = "\n\n".join([doc.page_content for doc in merged_docs])
        
        prompt = PromptTemplate(
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

RESPUESTA:
""",
            input_variables=["context", "question"],
        )
        
        llm = get_llm()
        llm_result = llm.invoke(prompt.format(context=context, question=good_query))
        answer = llm_result.content.strip()
        
        # 5. Verificar calidad de la respuesta
        verification = verify_answer_quality(good_query, answer, merged_docs)
        
        # Si se generó una respuesta mejorada, usarla
        final_answer = verification["improved_answer"] if verification["improved_answer"] else answer
        
        # Agregar nota si la respuesta es parcial
        if verification["quality"] == "partial" and not verification["improved_answer"]:
            final_answer += "\n\nNota: Esta respuesta podría ser parcial ya que no toda la información necesaria está disponible en el manual."
        
        # 6. Formatear la respuesta
        response = ChatResponse(
            original_query=q.query,
            rewritten_query=good_query,
            answer=final_answer,
            sources=[
                Source(
                    file=d.metadata.get("source", "desconocido"),
                    snippet=d.page_content[:200] + "…" if len(d.page_content) > 200 else d.page_content,
                )
                for d in merged_docs
            ],
        )
        
        process_time = time.time() - start_time
        logger.info(f"Consulta procesada en {process_time:.2f}s con calidad: {verification['quality']}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error al procesar consulta: {str(e)}", exc_info=True)
        # Manejo de errores (igual que antes)
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