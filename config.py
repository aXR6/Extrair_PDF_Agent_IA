#config.py
import os
from dotenv import load_dotenv
import logging

load_dotenv()

# === APIs e bancos de dados ===
OLLAMA_API_URL           = os.getenv('OLLAMA_API_URL', 'http://localhost:11434/api/embeddings')
OLLAMA_EMBEDDING_MODEL   = os.getenv('OLLAMA_EMBEDDING_MODEL', 'mxbai-embed-large')
MONGO_URI                = os.getenv('MONGO_URI')
DB_NAME                  = os.getenv('DB_NAME', 'ollama_chat')
COLL_PDF                 = os.getenv('COLL_PDF', 'PDF_')
COLL_BIN                 = os.getenv('COLL_BIN', 'Arq_PDF')
GRIDFS_BUCKET            = os.getenv('GRIDFS_BUCKET', 'fs')
PG_HOST                  = os.getenv('PG_HOST')
PG_PORT                  = os.getenv('PG_PORT')
PG_DB                    = os.getenv('PG_DB')
PG_USER                  = os.getenv('PG_USER')
PG_PASSWORD              = os.getenv('PG_PASSWORD')
PG_SCHEMA                = os.getenv('PG_SCHEMA', 'public')

# === Parâmetros de chunking e OCR ===
OCR_THRESHOLD            = int(os.getenv('OCR_THRESHOLD', '100'))
EMBEDDING_DIM            = int(os.getenv('EMBEDDING_DIM', '1024'))
CHUNK_SIZE               = int(os.getenv('CHUNK_SIZE', '1000'))
CHUNK_OVERLAP            = int(os.getenv('CHUNK_OVERLAP', '700'))

# === Separadores customizáveis ===
_sep_env = os.getenv('SEPARATORS')
SEPARATORS               = _sep_env.split('|') if _sep_env else ['\n\n','\n','.','!','?',';']
OCR_LANGUAGES            = os.getenv('OCR_LANGUAGES', 'eng+por')

# === Avançado: sliding window e semantic chunking ===
SLIDING_WINDOW_OVERLAP_RATIO = float(os.getenv('SLIDING_WINDOW_OVERLAP_RATIO', '0.2'))
SIMILARITY_THRESHOLD         = float(os.getenv('SIMILARITY_THRESHOLD', '0.8'))

# === Validação básica de variáveis críticas ===
def validate_config():
    missing = [k for k in ('MONGO_URI', 'PG_HOST') if not globals().get(k)]
    if missing:
        logging.warning('Variáveis críticas faltando: {}'.format(', '.join(missing)))
    for name, val in (('EMBEDDING_DIM', EMBEDDING_DIM), ('CHUNK_SIZE', CHUNK_SIZE), ('CHUNK_OVERLAP', CHUNK_OVERLAP)):
        if val <= 0:
            logging.error('{} inválido ({})'.format(name, val))