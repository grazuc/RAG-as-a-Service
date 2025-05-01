"""
ingest.py
----------
Ingiere documentos en múltiples formatos desde ./docs, los trocea,
genera embeddings y los guarda en la base de datos vectorial.
Soporta procesamiento paralelo y control de calidad.
"""

import os
import time
import hashlib
import argparse
import logging
from typing import List, Dict, Optional, Set, Tuple
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field

import numpy as np
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
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.pgvector import PGVector

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
    embed_model: str = "intfloat/multilingual-e5-base"
    collection_name: str = "manual_e5_multi5"
    chunk_size: int = 500
    chunk_overlap: int = 80
    batch_size: int = 100
    max_workers: int = 4
    allowed_languages: Set[str] = field(default_factory=lambda: {"es", "en"})
    file_extensions: List[str] = field(default_factory=lambda: ["pdf", "txt", "docx", "md"])
    deduplication: bool = True
    similarity_threshold: float = 0.95  # Para deduplicación

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
    parser.add_argument("--model", type=str, default="intfloat/multilingual-e5-base",
                        help="Modelo de embeddings a utilizar")
    parser.add_argument("--collection", type=str, default="manual_e5_multi5",
                        help="Nombre de la colección en PGVector")
    parser.add_argument("--chunk-size", type=int, default=500,
                        help="Tamaño de los chunks de texto")
    parser.add_argument("--chunk-overlap", type=int, default=80,
                        help="Superposición entre chunks")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Tamaño de lote para inserción en BD")
    parser.add_argument("--workers", type=int, default=4,
                        help="Número de workers para procesamiento paralelo")
    parser.add_argument("--langs", type=str, default="es,en",
                        help="Idiomas permitidos (códigos ISO separados por comas)")
    parser.add_argument("--extensions", type=str, default="pdf,txt,docx,md",
                        help="Extensiones de archivo a procesar (separadas por comas)")
    parser.add_argument("--no-dedup", action="store_true",
                        help="Desactivar deduplicación de chunks")
    
    args = parser.parse_args()
    
    # Crear configuración
    config = IngestConfig(
        docs_dir=Path(args.docs_dir),
        embed_model=args.model,
        collection_name=args.collection,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.batch_size,
        max_workers=args.workers,
        allowed_languages=set(args.langs.split(",")),
        file_extensions=args.extensions.split(","),
        deduplication=not args.no_dedup
    )
    
    return config, pg_conn

def get_file_loader(file_path: Path):
    """Selecciona el loader adecuado según la extensión del archivo"""
    extension = file_path.suffix.lower().lstrip(".")
    
    if extension == "pdf":
        return PyPDFLoader(str(file_path))
    elif extension == "docx":
        return Docx2txtLoader(str(file_path))
    elif extension in ["txt", "md"]:
        return TextLoader(str(file_path))
    else:
        # Para otros formatos, usar el loader genérico
        return UnstructuredFileLoader(str(file_path))

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

def process_file(file_path: Path, config: IngestConfig) -> List[Document]:
    """Procesa un archivo y devuelve los chunks resultantes"""
    try:
        logger.info(f"Procesando {file_path.name}")
        loader = get_file_loader(file_path)
        docs = loader.load()
        
        # Añadir metadatos
        for d in docs:
            d.metadata["source"] = file_path.name
            d.metadata["extension"] = file_path.suffix.lower().lstrip(".")
            d.metadata["ingest_time"] = time.time()
        
        # Dividir en chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(docs)
        
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
            
        return chunks
    except Exception as e:
        logger.error(f"Error procesando {file_path}: {e}")
        return []

def deduplicate_chunks(chunks: List[Document]) -> List[Document]:
    """Elimina chunks duplicados o muy similares basándose en sus hashes"""
    seen_hashes = set()
    unique_chunks = []
    
    for chunk in chunks:
        chunk_hash = chunk.metadata.get("content_hash", "")
        if chunk_hash and chunk_hash not in seen_hashes:
            seen_hashes.add(chunk_hash)
            unique_chunks.append(chunk)
    
    return unique_chunks

def chunks_to_batches(chunks: List[Document], batch_size: int) -> List[List[Document]]:
    """Divide la lista de chunks en lotes del tamaño especificado"""
    return [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]

def main():
    """Función principal de ingesta"""
    try:
        # Cargar configuración
        config, pg_conn = load_config()
        logger.info(f"Configuración cargada: {config}")
        
        print(f"🔍 Buscando documentos en {config.docs_dir.resolve()}")
        files_to_process = []
        for ext in config.file_extensions:
            files_to_process.extend(list(config.docs_dir.glob(f"*.{ext}")))
        
        if not files_to_process:
            raise FileNotFoundError(f"⚠️  No hay archivos con extensiones {config.file_extensions} en {config.docs_dir}")
        
        print(f"📄 Encontrados {len(files_to_process)} documentos")
        
        # Inicializar modelo de embeddings
        print(f"🧠 Inicializando modelo de embeddings: {config.embed_model}")
        embeddings = HuggingFaceEmbeddings(
            model_name=config.embed_model,
            model_kwargs={"device": "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu"},
            cache_folder="./model_cache",
        )
        
        # Procesar archivos en paralelo
        all_chunks = []
        with ProcessPoolExecutor(max_workers=config.max_workers) as executor:
            futures = {
                executor.submit(process_file, file_path, config): file_path
                for file_path in files_to_process
            }
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Procesando archivos"):
                file_path = futures[future]
                try:
                    chunks = future.result()
                    print(f"✂️  {file_path.name}: {len(chunks)} segmentos")
                    all_chunks.extend(chunks)
                except Exception as e:
                    logger.error(f"Error al procesar {file_path}: {e}")
        
        # Deduplicación si está activada
        if config.deduplication and all_chunks:
            original_count = len(all_chunks)
            all_chunks = deduplicate_chunks(all_chunks)
            print(f"🔄 Deduplicación: {original_count} → {len(all_chunks)} chunks")
        
        if not all_chunks:
            print("⚠️ No se generaron chunks válidos para insertar")
            return
            
        print(f"✅ Total de segmentos a insertar: {len(all_chunks)}")
        
        # Insertar por lotes
        batches = chunks_to_batches(all_chunks, config.batch_size)
        print(f"💾 Insertando {len(batches)} lotes en '{config.collection_name}'")
        
        for i, batch in enumerate(tqdm(batches, desc="Insertando lotes")):
            try:
                PGVector.from_documents(
                    documents=batch,
                    embedding=embeddings,
                    collection_name=config.collection_name,
                    connection_string=pg_conn,
                    pre_delete_collection=False,  # Importante: no borrar colección existente
                )
            except Exception as e:
                logger.error(f"Error al insertar lote {i+1}: {e}")
                
        print(f"✅ Ingesta completada en '{config.collection_name}'")
        print(f"📊 Resumen: {len(all_chunks)} segmentos insertados de {len(files_to_process)} archivos")
        
    except Exception as e:
        logger.error(f"Error en la ingesta: {e}")
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())