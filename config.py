# config.py
import os
import logging
from dotenv import load_dotenv

# Carrega variáveis de ambiente do .env
load_dotenv()

# ──────────────────────────────────────────────────────────────────────────────
# MongoDB
# ──────────────────────────────────────────────────────────────────────────────
MONGO_URI      = os.getenv("MONGO_URI")
DB_NAME        = os.getenv("DB_NAME", "ollama_chat")
COLL_PDF       = os.getenv("COLL_PDF", "PDF_")
COLL_BIN       = os.getenv("COLL_BIN", "Arq_PDF")
GRIDFS_BUCKET  = os.getenv("GRIDFS_BUCKET", "fs")

# ──────────────────────────────────────────────────────────────────────────────
# PostgreSQL
# ──────────────────────────────────────────────────────────────────────────────
PG_HOST        = os.getenv("PG_HOST")
PG_PORT        = os.getenv("PG_PORT")
PG_DB          = os.getenv("PG_DB")
PG_USER        = os.getenv("PG_USER")
PG_PASSWORD    = os.getenv("PG_PASSWORD")

# ──────────────────────────────────────────────────────────────────────────────
# Embeddings  
# ──────────────────────────────────────────────────────────────────────────────
OLLAMA_EMBEDDING_MODEL    = os.getenv("OLLAMA_EMBEDDING_MODEL", "mxbai-embed-large")
SERAFIM_EMBEDDING_MODEL   = os.getenv("SERAFIM_EMBEDDING_MODEL", "bigscience/serafim-900m")
MINILM_L6_V2              = os.getenv("MINILM_L6_V2", "sentence-transformers/all-MiniLM-L6-v2")
MINILM_L12_V2             = os.getenv("MINILM_L12_V2", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# ──────────────────────────────────────────────────────────────────────────────
# Embeddings dimensions
# ──────────────────────────────────────────────────────────────────────────────
DIM_MXBAI      = int(os.getenv("DIM_MXBAI", "1024"))
DIM_SERAFIM    = int(os.getenv("DIM_SERAFIM", "1536"))
DIM_MINILM_L6  = int(os.getenv("DIM_MINILM_L6", "384"))
DIM_MINIL12    = int(os.getenv("DIM_MINIL12", "384"))

# ──────────────────────────────────────────────────────────────────────────────
# Parâmetros de extração OCR
# ──────────────────────────────────────────────────────────────────────────────
OCR_THRESHOLD = int(os.getenv("OCR_THRESHOLD", "100"))

# ──────────────────────────────────────────────────────────────────────────────
# Parâmetros de chunking
# ──────────────────────────────────────────────────────────────────────────────
CHUNK_SIZE     = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP  = int(os.getenv("CHUNK_OVERLAP", str(CHUNK_SIZE // 2)))
SEPARATORS     = os.getenv("SEPARATORS", "\n\n|\n|.|!|?|;").split("|")

# ──────────────────────────────────────────────────────────────────────────────
# Idiomas para OCR fallback
# ──────────────────────────────────────────────────────────────────────────────
OCR_LANGUAGES = os.getenv("OCR_LANGUAGES", "eng+por")

# ──────────────────────────────────────────────────────────────────────────────
# Sliding-window e semantic chunking
# ──────────────────────────────────────────────────────────────────────────────
SLIDING_WINDOW_OVERLAP_RATIO = float(os.getenv("SLIDING_WINDOW_OVERLAP_RATIO", "0.25"))
SIMILARITY_THRESHOLD         = float(os.getenv("SIMILARITY_THRESHOLD", "0.8"))

# ──────────────────────────────────────────────────────────────────────────────
# Limite de tokens para modelos SBERT
# ──────────────────────────────────────────────────────────────────────────────
MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "128"))

# ──────────────────────────────────────────────────────────────────────────────
# Modelo SBERT (para semantic chunking) — remove referências ao antigo jvanhoof
# ──────────────────────────────────────────────────────────────────────────────
SBERT_MODEL_NAME = os.getenv(
    "SBERT_MODEL_NAME",
    "sentence-transformers/paraphrase-MiniLM-L6-v2"
)

# ──────────────────────────────────────────────────────────────────────────────
# Validação básica de variáveis críticas
# ──────────────────────────────────────────────────────────────────────────────
def validate_config():
    missing = [k for k in ("MONGO_URI", "PG_HOST") if not globals().get(k)]
    if missing:
        logging.warning(f"Variáveis críticas faltando: {', '.join(missing)}")
    for name, val in (("CHUNK_SIZE", CHUNK_SIZE), ("CHUNK_OVERLAP", CHUNK_OVERLAP)):
        if val <= 0:
            logging.error(f"{name} inválido ({val})")