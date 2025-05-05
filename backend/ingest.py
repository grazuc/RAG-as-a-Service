"""
ingest.py
----------
Ingiere documentos en múltiples formatos desde ./docs, los trocea,
genera embeddings y los guarda en la base de datos vectorial.
Soporta procesamiento paralelo, procesamiento incremental y chunking dinámico.
"""

import os
import time
import hashlib
import argparse
import logging
import json
import re
from typing import List, Dict, Optional, Set, Tuple, Any
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from langchain_huggingface import HuggingFaceEmbeddings
import multiprocessing
import shutil
import pickle
from datetime import datetime
import statistics

import numpy as np
import torch
from dotenv import load_dotenv
from tqdm import tqdm
from langdetect import detect, LangDetectException

from langchain_community.document_loaders import (
    PyPDFLoader, 
    UnstructuredFileLoader,
    DirectoryLoader,
    TextLoader, 
    Docx2txtLoader
)
from langchain_community.document_loaders.pdf import PyPDFium2Loader
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter
)
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.pgvector import PGVector
from threading import Lock

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("ingest.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuración por defecto
@dataclass
class IngestConfig:
    docs_dir: Path = Path("docs")
    cache_dir: Path = Path(".cache")
    embed_model: str = "intfloat/multilingual-e5-base" 
    collection_name: str = "manual_e5_multi"
    chunk_size: int = 384  # Chunk size base
    chunk_overlap: int = 64  # Chunk overlap base
    batch_size: int = 128  # Batch size más grande para mejor rendimiento
    max_workers: int = None  # Permitir autodetección
    max_embed_workers: int = 4  # Workers para embeddings
    allowed_languages: Set[str] = field(default_factory=lambda: {"es", "en"})
    file_extensions: List[str] = field(default_factory=lambda: ["pdf", "txt", "docx", "md"])
    deduplication: bool = True
    similarity_threshold: float = 0.95  # Para deduplicación
    incremental: bool = True  # Procesamiento incremental de archivos
    semantic_chunking: bool = True  # Chunking semántico
    dynamic_chunking: bool = True  # Chunking dinámico adaptativo


# 1) Entorno
def load_config() -> Tuple[IngestConfig, str]:
    """Carga la configuración desde argumentos y env vars"""
    load_dotenv()
    
    # Verificar variable de entorno obligatoria
    pg_conn = os.environ.get("PG_CONN")
    if not pg_conn:
        raise EnvironmentError("Variable de entorno PG_CONN no definida")
    
    # Parsear argumentos de línea de comandos
    parser = argparse.ArgumentParser(description="Ingesta de documentos para RAG")
    parser.add_argument("--docs-dir", type=str, default="docs", 
                        help="Directorio con documentos a ingerir")
    parser.add_argument("--cache-dir", type=str, default=".cache",
                        help="Directorio para caché de procesamiento")
    parser.add_argument("--model", type=str, default="intfloat/multilingual-e5-base",
                        help="Modelo de embeddings a utilizar")
    parser.add_argument("--collection", type=str, default="manual_e5_multi",
                        help="Nombre de la colección en PGVector")
    parser.add_argument("--chunk-size", type=int, default=384,
                        help="Tamaño base de los chunks de texto")
    parser.add_argument("--chunk-overlap", type=int, default=64,
                        help="Superposición base entre chunks")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Tamaño de lote para inserción en BD")
    parser.add_argument("--workers", type=int, default=None,
                        help="Número de workers para procesamiento paralelo (None=auto)")
    parser.add_argument("--embed-workers", type=int, default=4,
                        help="Número de workers para embeddings paralelos")
    parser.add_argument("--langs", type=str, default="es,en",
                        help="Idiomas permitidos (códigos ISO separados por comas)")
    parser.add_argument("--extensions", type=str, default="pdf,txt,docx,md",
                        help="Extensiones de archivo a procesar (separadas por comas)")
    parser.add_argument("--no-dedup", action="store_true",
                        help="Desactivar deduplicación de chunks")
    parser.add_argument("--no-incremental", action="store_true",
                        help="Desactivar procesamiento incremental")
    parser.add_argument("--no-semantic", action="store_true",
                        help="Desactivar chunking semántico")
    parser.add_argument("--no-dynamic", action="store_true",
                        help="Desactivar chunking dinámico adaptativo")
    parser.add_argument("--reset-cache", action="store_true",
                        help="Eliminar caché y reprocesar todos los archivos")
    
    args = parser.parse_args()
    
    # Auto-detección del número de workers si no se especifica
    max_workers = args.workers if args.workers is not None else max(1, multiprocessing.cpu_count() - 1)
    
    # Crear configuración
    config = IngestConfig(
        docs_dir=Path(args.docs_dir),
        cache_dir=Path(args.cache_dir),
        embed_model=args.model,
        collection_name=args.collection,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.batch_size,
        max_workers=max_workers,
        max_embed_workers=args.embed_workers,
        allowed_languages=set(args.langs.split(",")),
        file_extensions=args.extensions.split(","),
        deduplication=not args.no_dedup,
        incremental=not args.no_incremental,
        semantic_chunking=not args.no_semantic,
        dynamic_chunking=not args.no_dynamic
    )
    
    # Resetear caché si se solicita
    if args.reset_cache and config.cache_dir.exists():
        shutil.rmtree(config.cache_dir)
    
    # Asegurar que existe el directorio de caché
    config.cache_dir.mkdir(exist_ok=True, parents=True)
    
    return config, pg_conn


class DocumentProcessor:
    """Clase para manejo de procesamiento de documentos con caché"""
    
    def __init__(self, config: IngestConfig):
        self.config = config
        self.cache_file = config.cache_dir / "processed_files.json"
        self.processed_files = self._load_processed_files()
        
    def _load_processed_files(self) -> Dict[str, Dict[str, Any]]:
        """Carga el registro de archivos procesados desde caché"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error al cargar caché: {e}")
                return {}
        return {}
    
    def _save_processed_files(self):
        """Guarda el registro de archivos procesados en caché"""
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.processed_files, f)
    
    def get_files_to_process(self) -> List[Tuple[Path, bool]]:
        """
        Devuelve lista de archivos a procesar, indicando si necesitan procesamiento
        Retorna: Lista de tuplas (archivo, necesita_procesamiento)
        """
        files_to_process = []
        
        for ext in self.config.file_extensions:
            for file_path in self.config.docs_dir.glob(f"*.{ext}"):
                # Obtener información del archivo
                file_key = file_path.name
                file_stat = file_path.stat()
                file_size = file_stat.st_size
                file_mtime = file_stat.st_mtime
                
                # Verificar si el archivo necesita procesamiento
                needs_processing = True
                if self.config.incremental and file_key in self.processed_files:
                    cached_info = self.processed_files[file_key]
                    # Si tamaño y fecha de modificación coinciden, no procesar
                    if cached_info.get("size") == file_size and cached_info.get("mtime") == file_mtime:
                        needs_processing = False
                
                files_to_process.append((file_path, needs_processing))
        
        return files_to_process
    
    def mark_file_processed(self, file_path: Path):
        """Marca un archivo como procesado en el caché"""
        file_stat = file_path.stat()
        self.processed_files[file_path.name] = {
            "size": file_stat.st_size,
            "mtime": file_stat.st_mtime,
            "last_processed": time.time()
        }
        self._save_processed_files()
    
    def get_chunks_cache_path(self, file_path: Path) -> Path:
        """Devuelve la ruta para caché de chunks de un archivo"""
        file_hash = hashlib.md5(str(file_path).encode()).hexdigest()
        return self.config.cache_dir / f"chunks_{file_hash}.pkl"
    
    def get_cached_chunks(self, file_path: Path) -> Optional[List[Document]]:
        """Intenta obtener chunks cacheados para un archivo"""
        if not self.config.incremental:
            return None
            
        cache_path = self.get_chunks_cache_path(file_path)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Error al cargar chunks desde caché: {e}")
        return None
    
    def save_chunks_to_cache(self, file_path: Path, chunks: List[Document]):
        """Guarda chunks en caché para un archivo"""
        if not self.config.incremental:
            return
            
        cache_path = self.get_chunks_cache_path(file_path)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(chunks, f)
        except Exception as e:
            logger.warning(f"Error al guardar chunks en caché: {e}")


def get_file_loader(file_path: Path, allowed_extensions):
    """Selecciona el loader más adecuado según el tipo de archivo"""
    extension = file_path.suffix.lower().lstrip(".")
    
    if extension not in allowed_extensions:
        raise ValueError(f"Extensión no soportada: {extension}")
    
    try:
        if extension == "pdf":
            # Usar PyPDFium2Loader para mejor rendimiento y extracción en PDFs
            return PyPDFium2Loader(str(file_path))
        elif extension == "docx":
            return Docx2txtLoader(str(file_path))
        elif extension == "md":
            return TextLoader(str(file_path), encoding='utf-8')
        elif extension == "txt":
            return TextLoader(str(file_path), encoding='utf-8')
        else:
            return UnstructuredFileLoader(str(file_path))
    except Exception as e:
        logger.error(f"Error al crear loader para {file_path}: {e}")
        raise


def detect_language(text: str) -> Optional[str]:
    """Detecta el idioma del texto, devuelve None si falla"""
    if not text or len(text.strip()) < 20:
        return None
    
    try:
        return detect(text)
    except LangDetectException:
        return None


def compute_text_hash(text: str) -> str:
    """Calcula un hash del texto para deduplicación"""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def analyze_document_structure(text: str, file_extension: str) -> Dict[str, Any]:
    """
    Analiza la estructura del documento para determinar características relevantes
    para el chunking dinámico.
    
    Retorna un diccionario con métricas del documento.
    """
    # Inicializar análisis
    analysis = {
        "extension": file_extension,
        "doc_type": "general",
        "avg_sentence_length": 0,
        "avg_paragraph_length": 0,
        "semantic_density": 0.0,
        "has_structures": False,
        "recommended_chunk_size": 384,  # Valor por defecto
        "recommended_chunk_overlap": 64  # Valor por defecto
    }
    
    # Detectar tipo de documento por extensión y patrones
    if file_extension == "pdf":
        analysis["doc_type"] = "pdf"
    elif file_extension == "docx":
        analysis["doc_type"] = "docx"
    elif file_extension == "md":
        analysis["doc_type"] = "markdown"
    elif file_extension == "txt":
        # Intentar identificar el tipo específico de documento de texto
        if re.search(r"(artículo|cláusula|contrato|acuerdo)", text.lower()):
            analysis["doc_type"] = "legal"
        elif re.search(r"(from:|to:|subject:|sent:)", text.lower()):
            analysis["doc_type"] = "email"
        elif re.search(r"(abstract|introducción|metodología|conclusión)", text.lower()):
            analysis["doc_type"] = "scientific"
    
    # Análisis de estructura
    paragraphs = re.split(r'\n\s*\n', text)
    if len(paragraphs) > 1:
        # Calcular longitud promedio de párrafos
        paragraph_lengths = [len(p.split()) for p in paragraphs if p.strip()]
        if paragraph_lengths:
            analysis["avg_paragraph_length"] = sum(paragraph_lengths) / len(paragraph_lengths)
    
    # Análisis de oraciones
    sentences = re.split(r'[.!?]+', text)
    if sentences:
        sentence_lengths = [len(s.split()) for s in sentences if s.strip()]
        if sentence_lengths:
            analysis["avg_sentence_length"] = sum(sentence_lengths) / len(sentence_lengths)
            
            # Calcular desviación estándar para complejidad del texto
            if len(sentence_lengths) > 1:
                std_dev = statistics.stdev(sentence_lengths)
                # Mayor desviación estándar sugiere mayor variabilidad en complejidad
                analysis["complexity"] = std_dev / analysis["avg_sentence_length"]
            else:
                analysis["complexity"] = 0
    
    # Estimación de densidad semántica
    # - Proporción de palabras significativas (excluyendo stopwords)
    # - Frecuencia de términos técnicos o especializados
    words = re.findall(r'\b\w+\b', text.lower())
    unique_words = set(words)
    
    # Ratio simple de diversidad léxica
    if words:
        lexical_diversity = len(unique_words) / len(words)
        analysis["semantic_density"] = lexical_diversity
    
    # Detectar estructuras especiales
    analysis["has_structures"] = bool(
        re.search(r'(tabla|cuadro|figura|gráfico|imagen|código|code|```)', text.lower()) or
        re.search(r'^\s*[\d.]+\s+\w+', text, re.MULTILINE)  # Listas numeradas
    )
    
    # Decidir tamaño de chunk y overlap recomendados según el análisis
    determine_chunking_parameters(analysis)
    
    return analysis


def determine_chunking_parameters(analysis: Dict[str, Any]):
    """
    Determina los parámetros óptimos de chunking basados en el análisis del documento.
    Modifica el diccionario de análisis in-place.
    """
    doc_type = analysis["doc_type"]
    
    # Base de chunking según tipo de documento
    if doc_type == "legal":
        # Documentos legales requieren chunks más grandes para mantener contexto legal
        base_chunk_size = 500
        base_chunk_overlap = 100
    elif doc_type == "scientific":
        # Documentos científicos con chunks medianos pero mayor overlap
        base_chunk_size = 450
        base_chunk_overlap = 90
    elif doc_type == "email":
        # Emails típicamente son más cortos
        base_chunk_size = 300
        base_chunk_overlap = 50
    elif doc_type == "markdown":
        # Documentos markdown estructurados
        base_chunk_size = 400
        base_chunk_overlap = 80
    else:
        # Valores base para otros tipos
        base_chunk_size = 384
        base_chunk_overlap = 64
    
    # Ajustes basados en densidad semántica
    density_factor = 1.0
    if analysis["semantic_density"] > 0.7:  # Alta diversidad léxica
        # Para texto denso, reducir tamaño del chunk
        density_factor = 0.8
    elif analysis["semantic_density"] < 0.4:  # Baja diversidad léxica
        # Para texto simple, aumentar tamaño del chunk
        density_factor = 1.2
    
    # Ajustes basados en longitud de oraciones
    sentence_factor = 1.0
    if analysis.get("avg_sentence_length", 0) > 25:  # Oraciones largas
        # Reducir tamaño para texto con oraciones complejas
        sentence_factor = 0.85
    elif analysis.get("avg_sentence_length", 0) < 10:  # Oraciones cortas
        # Aumentar tamaño para texto con oraciones simples
        sentence_factor = 1.15
    
    # Ajuste si el documento tiene estructuras especiales
    structure_factor = 0.9 if analysis["has_structures"] else 1.0
    
    # Calcular tamaños finales ajustados
    chunk_size = int(base_chunk_size * density_factor * sentence_factor * structure_factor)
    
    # Ajustar overlap: mayor complejidad -> mayor overlap
    complexity_factor = 1.0 + (analysis.get("complexity", 0) * 0.5)
    chunk_overlap = int(base_chunk_overlap * complexity_factor)
    
    # Restricciones para mantener valores razonables
    chunk_size = max(200, min(chunk_size, 1000))  # Entre 200 y 1000
    chunk_overlap = max(32, min(chunk_overlap, chunk_size // 2))  # Entre 32 y 50% del chunk_size
    
    # Actualizar el análisis con los valores recomendados
    analysis["recommended_chunk_size"] = chunk_size
    analysis["recommended_chunk_overlap"] = chunk_overlap


def get_smart_text_splitter(doc_extension: str, config: IngestConfig, doc_content: Optional[str] = None):
    """
    Devuelve el splitter más adecuado según el tipo de documento y su contenido
    """
    # Usar chunking dinámico si está activado y se proporciona contenido
    if config.dynamic_chunking and doc_content:
        # Analizar el documento para determinar los mejores parámetros
        doc_analysis = analyze_document_structure(doc_content, doc_extension)
        
        # Usar los parámetros recomendados
        chunk_size = doc_analysis["recommended_chunk_size"]
        chunk_overlap = doc_analysis["recommended_chunk_overlap"]
        
        logger.info(f"Chunking dinámico: size={chunk_size}, overlap={chunk_overlap} para doc tipo {doc_analysis['doc_type']}")
    else:
        # Usar valores por defecto si no hay chunking dinámico
        chunk_size = config.chunk_size
        chunk_overlap = config.chunk_overlap
    
    # Si no se usa chunking semántico, usar RecursiveCharacterTextSplitter simple
    if not config.semantic_chunking:
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
    
    # Personalizar el splitter según el tipo de documento
    if doc_extension == "md":
        # Para archivos Markdown, usar splitter basado en encabezados
        headers_to_split_on = [
            ("#", "Heading1"),
            ("##", "Heading2"),
            ("###", "Heading3"),
            ("####", "Heading4"),
        ]
        
        # Primero dividir por encabezados
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        
        # Luego dividir por tamaño con separadores inteligentes
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
    
    elif doc_extension == "pdf" or doc_extension == "docx":
        # Para PDF y DOCX, usar separadores que respeten estructura de documento
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n\n",  # Separación entre secciones principales
                "\n\n",    # Separación entre párrafos
                "\n",      # Separación entre líneas
                ". ",      # Separación entre frases
                "; ",      # Separación dentro de frases complejas
                ", ",      # Separación dentro de frases
                " ",       # Separación entre palabras
                ""         # Último recurso: caracteres individuales
            ],
        )
    else:
        # Para otros tipos, usar configuración estándar
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )


def process_file(file_path: Path, config: IngestConfig, doc_processor: DocumentProcessor) -> List[Document]:
    """Procesa un archivo y devuelve los chunks resultantes"""
    try:
        logger.info(f"Procesando {file_path.name}")
        
        # Verificar si podemos usar caché
        if not doc_processor.get_cached_chunks(file_path):
            loader = get_file_loader(file_path, config.file_extensions)
            docs = loader.load()
            
            # Añadir metadatos
            file_extension = file_path.suffix.lower().lstrip(".")
            for d in docs:
                d.metadata["source"] = file_path.name
                d.metadata["extension"] = file_extension
                d.metadata["ingest_time"] = time.time()
                d.metadata["chunk_id"] = 0  # Inicializar chunk_id
                
                # Añadir metadatos avanzados
                d.metadata["file_size"] = file_path.stat().st_size
                d.metadata["doc_type"] = file_extension
                d.metadata["process_date"] = datetime.now().isoformat()
            
            # Si es un solo documento, usar todo el contenido para análisis
            full_content = docs[0].page_content if len(docs) == 1 else "\n\n".join([d.page_content for d in docs])
            
            # Dividir en chunks usando el splitter adecuado con chunking dinámico
            splitter = get_smart_text_splitter(file_extension, config, full_content)
            chunks = splitter.split_documents(docs)
            
            # Añadir chunk_id incremental para relacionar chunks consecutivos
            for i, chunk in enumerate(chunks):
                chunk.metadata["chunk_id"] = i
                chunk.metadata["chunk_index"] = i
                chunk.metadata["chunk_total"] = len(chunks)
                
                # Preservar contexto del chunk anterior/siguiente para mejor recuperación
                if i > 0:
                    # Añadir referencia al chunk anterior
                    chunk.metadata["prev_chunk_id"] = i - 1
                if i < len(chunks) - 1:
                    # Añadir referencia al chunk siguiente
                    chunk.metadata["next_chunk_id"] = i + 1
                    
                # Si se usó chunking dinámico, añadir metadatos de configuración de chunking
                if config.dynamic_chunking:
                    doc_analysis = analyze_document_structure(full_content, file_extension)
                    chunk.metadata["chunk_size_used"] = doc_analysis["recommended_chunk_size"]
                    chunk.metadata["chunk_overlap_used"] = doc_analysis["recommended_chunk_overlap"]
                    chunk.metadata["doc_semantic_density"] = doc_analysis["semantic_density"]
                    chunk.metadata["doc_analysis"] = {
                        k: v for k, v in doc_analysis.items() 
                        if k not in ["recommended_chunk_size", "recommended_chunk_overlap"]
                    }
            
            # Filtrar por idioma
            if config.allowed_languages:
                filtered_chunks = []
                for chunk in chunks:
                    lang = detect_language(chunk.page_content)
                    if lang and lang in config.allowed_languages:
                        chunk.metadata["language"] = lang
                        filtered_chunks.append(chunk)
                    else:
                        logger.debug(f"Chunk descartado por idioma: {lang}")
                chunks = filtered_chunks
                
            # Añadir hash para deduplicación
            for chunk in chunks:
                chunk.metadata["content_hash"] = compute_text_hash(chunk.page_content)
                
            # Guardar chunks en caché
            doc_processor.save_chunks_to_cache(file_path, chunks)
            
            # Marcar archivo como procesado
            doc_processor.mark_file_processed(file_path)
            
            return chunks
        else:
            # Usar chunks cacheados
            logger.info(f"Usando chunks cacheados para {file_path.name}")
            chunks = doc_processor.get_cached_chunks(file_path)
            
            # Actualizar algunos metadatos que pueden cambiar
            for chunk in chunks:
                chunk.metadata["ingest_time"] = time.time()
                
            return chunks
            
    except Exception as e:
        logger.error(f"Error procesando {file_path}: {e}")
        return []


# Lock global para deduplicación
dedup_lock = Lock()

def deduplicate_chunks(chunks: List[Document], similarity_threshold=0.95) -> List[Document]:
    """Elimina chunks duplicados o muy similares con sincronización"""
    with dedup_lock:
        seen_hashes = set()
        content_dict = {}  # Para almacenar textos y detectar similitudes
        unique_chunks = []
        
        for chunk in chunks:
            chunk_hash = chunk.metadata.get("content_hash", "")
            content = chunk.page_content.strip()
            
            # Si el hash ya existe, es un duplicado exacto
            if chunk_hash and chunk_hash in seen_hashes:
                continue
                
            # Marcamos como visto
            if chunk_hash:
                seen_hashes.add(chunk_hash)
                
            # Añadimos a chunks únicos
            unique_chunks.append(chunk)
    
    return unique_chunks


def batch_process_embeddings(chunks: List[Document], embeddings, batch_size: int) -> List[Tuple[Document, List[float]]]:
    """
    Procesa embeddings en batch para mejor rendimiento
    Retorna lista de tuplas (documento, embedding)
    """
    if not chunks:
        return []
    
    # Dividir en lotes para embedding eficiente
    chunks_batches = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]
    results = []
    
    for batch in chunks_batches:
        # Extraer textos
        texts = [doc.page_content for doc in batch]
        
        # Calcular embeddings para todo el lote
        batch_embeddings = embeddings.embed_documents(texts)
        
        # Asociar cada documento con su embedding
        for i, doc in enumerate(batch):
            results.append((doc, batch_embeddings[i]))
    
    return results


def parallel_embed_chunks(chunks: List[Document], embeddings, batch_size: int, workers: int) -> List[Tuple[Document, List[float]]]:
    """
    Calcula embeddings en paralelo usando múltiples workers
    Retorna lista de tuplas (documento, embedding)
    """
    if not chunks:
        return []
    
    # Dividir en bloques de trabajo para cada worker
    chunk_blocks = [chunks[i:i + batch_size*2] for i in range(0, len(chunks), batch_size*2)]
    results = []
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(batch_process_embeddings, block, embeddings, batch_size): i
            for i, block in enumerate(chunk_blocks)
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Calculando embeddings"):
            try:
                batch_results = future.result()
                results.extend(batch_results)
            except Exception as e:
                logger.error(f"Error en cálculo de embeddings: {e}")
                
    return results


def insert_into_pgvector(doc_embed_pairs: List[Tuple[Document, List[float]]], pg_conn: str, collection_name: str, embeddings):
    """Inserta documentos y sus embeddings en la base de datos PGVector"""
    if not doc_embed_pairs:
        logger.info("No hay documentos para insertar")
        return
    
    # Crear conexión al vector store
    try:
        vector_store = PGVector.from_documents(
            documents=[],  # Inicialmente vacío, insertaremos por lotes
            embedding=embeddings,  # No necesitamos el embedder aquí
            collection_name=collection_name,
            connection_string=pg_conn,
            pre_delete_collection=True  
        )
        
        # Preparar documentos con embeddings para inserción
        embeddings_list = []
        docs = []
        
        for doc, embedding in doc_embed_pairs:
            docs.append(doc)
            embeddings_list.append(embedding)
        
        # Insertar en la base de datos - corregido para usar el formato esperado por PGVector
        texts = [d.page_content for d in docs]
        metadatas = [d.metadata for d in docs]
        
        vector_store.add_embeddings(
            texts=texts,
            embeddings=embeddings_list,
            metadatas=metadatas
        )
        
        logger.info(f"Insertados {len(docs)} documentos en la colección {collection_name}")
        
    except Exception as e:
        logger.error(f"Error al insertar en PGVector: {e}")
        raise


def main():
    """Función principal del proceso de ingesta"""
    # Cargar configuración
    config, pg_conn = load_config()
    logger.info(f"Configuración cargada: {vars(config)}")
    
    # Inicializar procesador de documentos
    doc_processor = DocumentProcessor(config)
    
    # Obtener lista de archivos a procesar
    files_to_process = doc_processor.get_files_to_process()
    logger.info(f"Total de archivos encontrados: {len(files_to_process)}")
    
    if not files_to_process:
        logger.info("No hay archivos para procesar")
        return
    
    # Filtrar solo los archivos que necesitan procesamiento
    files_needing_processing = [(f, n) for f, n in files_to_process if n]
    logger.info(f"Archivos que necesitan procesamiento: {len(files_needing_processing)}")
    
    if not files_needing_processing:
        logger.info("Todos los archivos ya están procesados")
        return
    
    # Procesar archivos en paralelo
    chunks_from_files = []
    
    with ProcessPoolExecutor(max_workers=config.max_workers) as executor:
        # Crear tareas de procesamiento
        futures = {
            executor.submit(process_file, file_path, config, doc_processor): file_path
            for file_path, needs_processing in files_needing_processing
        }
        
        # Recoger resultados a medida que se completan
        for future in tqdm(as_completed(futures), total=len(futures), desc="Procesando archivos"):
            file_path = futures[future]
            try:
                file_chunks = future.result()
                if file_chunks:
                    chunks_from_files.extend(file_chunks)
                    logger.info(f"Procesado {file_path.name}: {len(file_chunks)} chunks")
                else:
                    logger.warning(f"No se obtuvieron chunks de {file_path.name}")
            except Exception as e:
                logger.error(f"Error procesando {file_path.name}: {e}")
    
    # Aplicar deduplicación si está configurada
    if config.deduplication and chunks_from_files:
        original_count = len(chunks_from_files)
        chunks_from_files = deduplicate_chunks(chunks_from_files, config.similarity_threshold)
        logger.info(f"Deduplicación: {original_count} -> {len(chunks_from_files)} chunks")
    
    # Si no hay chunks para procesar, terminar
    if not chunks_from_files:
        logger.info("No hay chunks para procesar después de la deduplicación")
        return
    
    # Inicializar modelo de embeddings
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=config.embed_model,
            model_kwargs={"device": "cuda"} if torch.cuda.is_available() else {"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        logger.info(f"Modelo de embeddings cargado: {config.embed_model}")
    except Exception as e:
        logger.error(f"Error al cargar el modelo de embeddings: {e}")
        return
    
    # Calcular embeddings en paralelo
    logger.info(f"Calculando embeddings para {len(chunks_from_files)} chunks...")
    doc_embed_pairs = parallel_embed_chunks(
        chunks_from_files, 
        embeddings, 
        config.batch_size,
        config.max_embed_workers
    )
    
    # Insertar en la base de datos vectorial
    logger.info("Insertando documentos en PGVector...")
    insert_into_pgvector(doc_embed_pairs, pg_conn, config.collection_name, embeddings)
    
    logger.info("Proceso de ingesta completado con éxito")


# Ejecutar programa principal si se llama directamente
if __name__ == "__main__":
    # Registrar tiempo de inicio
    start_time = time.time()
    
    # Ejecutar proceso principal
    try:
        main()
    except Exception as e:
        logger.error(f"Error en el proceso principal: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    # Registrar tiempo de finalización
    elapsed_time = time.time() - start_time
    logger.info(f"Tiempo total de ejecución: {elapsed_time:.2f} segundos")