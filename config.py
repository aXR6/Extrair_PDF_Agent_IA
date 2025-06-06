#config.py
import os
import logging
from dotenv import load_dotenv

load_dotenv()

# — NVD API Key (para incremental)
NVD_API_KEY = os.getenv("NVD_API_KEY")

# — PostgreSQL Connection
PG_HOST     = os.getenv("PG_HOST")
PG_PORT     = int(os.getenv("PG_PORT", "5432"))
PG_USER     = os.getenv("PG_USER")
PG_PASSWORD = os.getenv("PG_PASSWORD")
PG_DATABASE = os.getenv("PG_DATABASE")

# — CSV locais (NVD)
CSV_FULL = os.getenv("CSV_FULL")
CSV_INCR = os.getenv("CSV_INCR")

# — Modelos de Embedding & Chunking
OLLAMA_EMBEDDING_MODEL  = os.getenv("OLLAMA_EMBEDDING_MODEL")
SERAFIM_EMBEDDING_MODEL = os.getenv("SERAFIM_EMBEDDING_MODEL")
MINILM_L6_V2            = os.getenv("MINILM_L6_V2")
MINILM_L12_V2           = os.getenv("MINILM_L12_V2")
MPNET_EMBEDDING_MODEL   = os.getenv("MPNET_EMBEDDING_MODEL")
SBERT_MODEL_NAME        = os.getenv("SBERT_MODEL_NAME")

# — Dimensões de embedding
DIM_MXBAI     = int(os.getenv("DIM_MXBAI", "0"))
DIM_SERAFIM   = int(os.getenv("DIM_SERAFIM", "0"))
DIM_MINILM_L6 = int(os.getenv("DIM_MINILM_L6", "0"))
DIM_MINIL12   = int(os.getenv("DIM_MINIL12", "0"))
DIM_MPNET     = int(os.getenv("DIM_MPNET", "0"))

# — Parâmetros OCR
OCR_THRESHOLD = int(os.getenv("OCR_THRESHOLD", "100"))
# Usa 'eng+por' como padrão se OCR_LANGUAGES não estiver definido
OCR_LANGUAGES = os.getenv("OCR_LANGUAGES", "eng+por")
# Timeout máximo para pdf2image.convert_from_path (segundos)
PDF2IMAGE_TIMEOUT = int(os.getenv("PDF2IMAGE_TIMEOUT", "600"))
# Parâmetros adicionais passados ao Tesseract (ex: "--oem 3 --psm 6")
TESSERACT_CONFIG = os.getenv("TESSERACT_CONFIG", "")

# — Parâmetros de chunking
CHUNK_SIZE                   = int(os.getenv("CHUNK_SIZE", "0"))
CHUNK_OVERLAP                = int(os.getenv("CHUNK_OVERLAP", "0"))
SLIDING_WINDOW_OVERLAP_RATIO = float(os.getenv("SLIDING_WINDOW_OVERLAP_RATIO", "0.0"))
MAX_SEQ_LENGTH               = int(os.getenv("MAX_SEQ_LENGTH", "0"))
SEPARATORS                   = os.getenv("SEPARATORS", "").split("|")

def validate_config():
    missing = []
    for var in ("PG_HOST", "PG_PORT", "PG_USER", "PG_PASSWORD", "PG_DATABASE"):
        if not globals().get(var):
            missing.append(var)
    if missing:
        logging.error(f"Variáveis críticas faltando: {missing}")
        raise RuntimeError(f"Variáveis faltando: {missing}")
