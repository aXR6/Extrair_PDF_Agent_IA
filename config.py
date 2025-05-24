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
PG_SCHEMA      = os.getenv("PG_SCHEMA", "public")

# ──────────────────────────────────────────────────────────────────────────────
# Embeddings Ollama (API)
# ──────────────────────────────────────────────────────────────────────────────
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "mxbai-embed-large")
OLLAMA_API_URL         = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/embeddings")

# ──────────────────────────────────────────────────────────────────────────────
# Embeddings Serafim
# ──────────────────────────────────────────────────────────────────────────────
SERAFIM_EMBEDDING_MODEL = os.getenv("SERAFIM_EMBEDDING_MODEL", "serafim-900m")
SERAFIM_DIM             = int(os.getenv("SERAFIM_DIM", "1536"))

# ──────────────────────────────────────────────────────────────────────────────
# Modelo SBERT (para semantic chunking)
# ──────────────────────────────────────────────────────────────────────────────
SBERT_MODEL_NAME = os.getenv(
    "SBERT_MODEL_NAME",
    "jvanhoof/all-MiniLM-L6-multilingual-v2-en-es-pt-pt-br-v2"
)

# ──────────────────────────────────────────────────────────────────────────────
# Parâmetros gerais
# ──────────────────────────────────────────────────────────────────────────────
OCR_THRESHOLD               = int(os.getenv("OCR_THRESHOLD", "100"))
EMBEDDING_DIM               = int(os.getenv("EMBEDDING_DIM", "1024"))
CHUNK_SIZE_ENV              = os.getenv("CHUNK_SIZE")
CHUNK_OVERLAP_ENV           = os.getenv("CHUNK_OVERLAP")
SLIDING_WINDOW_OVERLAP_RATIO = float(os.getenv("SLIDING_WINDOW_OVERLAP_RATIO", "0.25"))
SIMILARITY_THRESHOLD         = float(os.getenv("SIMILARITY_THRESHOLD", "0.8"))
MAX_SEQ_LENGTH               = int(os.getenv("MAX_SEQ_LENGTH", "128"))

# ──────────────────────────────────────────────────────────────────────────────
# Chunk Size e Overlap
# ──────────────────────────────────────────────────────────────────────────────
if CHUNK_SIZE_ENV:
    CHUNK_SIZE = int(CHUNK_SIZE_ENV)
else:
    CHUNK_SIZE = 512

if CHUNK_OVERLAP_ENV:
    CHUNK_OVERLAP = int(CHUNK_OVERLAP_ENV)
else:
    CHUNK_OVERLAP = CHUNK_SIZE // 2

# ──────────────────────────────────────────────────────────────────────────────
# Separadores customizáveis
# ──────────────────────────────────────────────────────────────────────────────
_sep_env   = os.getenv("SEPARATORS")
SEPARATORS = _sep_env.split("|") if _sep_env else ["\n\n", "\n", ".", "!", "?", ";"]

# ──────────────────────────────────────────────────────────────────────────────
# Idiomas para OCR fallback
# ──────────────────────────────────────────────────────────────────────────────
OCR_LANGUAGES = os.getenv("OCR_LANGUAGES", "eng+por")

# ──────────────────────────────────────────────────────────────────────────────
# Validação básica
# ──────────────────────────────────────────────────────────────────────────────
def validate_config():
    missing = [k for k in ("MONGO_URI", "PG_HOST") if not globals().get(k)]
    if missing:
        logging.warning(f"Variáveis críticas faltando: {', '.join(missing)}")
    for name, val in (("CHUNK_SIZE", CHUNK_SIZE), ("CHUNK_OVERLAP", CHUNK_OVERLAP)):
        if val <= 0:
            logging.error(f"{name} inválido ({val})")
