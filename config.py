# =====================================================================
# config.py
# =====================================================================
import os
import logging
from dotenv import load_dotenv

load_dotenv()

# NVD
NVD_API_KEY = os.getenv("NVD_API_KEY")

# PostgreSQL
PG_HOST       = os.getenv("PG_HOST")
PG_PORT       = int(os.getenv("PG_PORT", "5432"))
PG_USER       = os.getenv("PG_USER")
PG_PASSWORD   = os.getenv("PG_PASSWORD")
# Schemas (lista) e default
PG_SCHEMAS         = [s.strip() for s in os.getenv("PG_SCHEMAS", "").split(",") if s.strip()]
PG_SCHEMA_DEFAULT  = os.getenv("PG_SCHEMA_DEFAULT")

# CSV locais
CSV_FULL = os.getenv("CSV_FULL")
CSV_INCR = os.getenv("CSV_INCR")

# Modelos de embedding e chunking
OLLAMA_EMBEDDING_MODEL  = os.getenv("OLLAMA_EMBEDDING_MODEL")
SERAFIM_EMBEDDING_MODEL = os.getenv("SERAFIM_EMBEDDING_MODEL")
MINILM_L6_V2            = os.getenv("MINILM_L6_V2")
MINILM_L12_V2           = os.getenv("MINILM_L12_V2")
MPNET_EMBEDDING_MODEL   = os.getenv("MPNET_EMBEDDING_MODEL")
SBERT_MODEL_NAME        = os.getenv("SBERT_MODEL_NAME")

# Dimensões
DIM_MXBAI     = int(os.getenv("DIM_MXBAI", "0"))
DIM_SERAFIM   = int(os.getenv("DIM_SERAFIM", "0"))
DIM_MINILM_L6 = int(os.getenv("DIM_MINILM_L6", "0"))
DIM_MINIL12   = int(os.getenv("DIM_MINIL12", "0"))
DIM_MPNET     = int(os.getenv("DIM_MPNET", "0"))

# OCR
OCR_THRESHOLD = int(os.getenv("OCR_THRESHOLD", "100"))
OCR_LANGUAGES = os.getenv("OCR_LANGUAGES")

# Chunking
CHUNK_SIZE                   = int(os.getenv("CHUNK_SIZE", "0"))
CHUNK_OVERLAP                = int(os.getenv("CHUNK_OVERLAP", "0"))
SLIDING_WINDOW_OVERLAP_RATIO = float(os.getenv("SLIDING_WINDOW_OVERLAP_RATIO", "0.0"))
MAX_SEQ_LENGTH               = int(os.getenv("MAX_SEQ_LENGTH", "0"))
SEPARATORS                   = os.getenv("SEPARATORS").split("|")

# Validação

def validate_config():
    missing = []
    for var in ("PG_HOST","PG_PORT","PG_USER","PG_PASSWORD"): 
        if not globals().get(var):
            missing.append(var)
    if not PG_SCHEMAS or not PG_SCHEMA_DEFAULT:
        missing.append("PG_SCHEMAS/PG_SCHEMA_DEFAULT")
    if missing:
        logging.error(f"Variáveis críticas faltando: {missing}")
        raise RuntimeError(f"Variáveis faltando: {missing}")