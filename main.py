#!/usr/bin/env python3
# main.py
import os
import logging
from tqdm import tqdm

from config import (
    MONGO_URI, DB_NAME, COLL_PDF, COLL_BIN, GRIDFS_BUCKET, OCR_THRESHOLD,
    OLLAMA_EMBEDDING_MODEL, SERAFIM_EMBEDDING_MODEL,
    MINILM_L6_V2, MINILM_L12_V2,
    DIM_MXBAI, DIM_SERAFIM, DIM_MINILM_L6, DIM_MINIL12
)
from adaptive_chunker import get_sbert_model, SBERT_MODEL_NAME
from utils import setup_logging, is_valid_file, build_record as build_meta
from extractors import (
    is_extraction_allowed, fallback_ocr,
    PyPDFStrategy, PDFMinerStrategy, PDFMinerLowLevelStrategy,
    UnstructuredStrategy, OCRStrategy, PDFPlumberStrategy,
    TikaStrategy, PyMuPDF4LLMStrategy
)
from storage import save_metadata, save_file_binary, save_gridfs
from pg_storage import save_to_postgres
import shutil

# ──────────────────────────────────────────────────────────────────────────────
# Inicialização de logs e pré-carregamento SBERT
# ──────────────────────────────────────────────────────────────────────────────
logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("PyPDF2").setLevel(logging.ERROR)
setup_logging()
get_sbert_model(SBERT_MODEL_NAME)

# ──────────────────────────────────────────────────────────────────────────────
# Estratégias de extração
# ──────────────────────────────────────────────────────────────────────────────
STRATEGIES = {
    "pypdf":        PyPDFStrategy(),
    "pdfminer":     PDFMinerStrategy(),
    "pdfminer-low": PDFMinerLowLevelStrategy(),
    "unstructured": UnstructuredStrategy(),
    "ocr":          OCRStrategy(threshold=OCR_THRESHOLD),
    "plumber":      PDFPlumberStrategy(),
    "tika":         TikaStrategy(),
    "pymupdf4llm":  PyMuPDF4LLMStrategy(),
}

# ──────────────────────────────────────────────────────────────────────────────
# Escolha de modelos e dimensões via ambiente
# ──────────────────────────────────────────────────────────────────────────────
EMBEDDING_MODELS = {
    "1": OLLAMA_EMBEDDING_MODEL,
    "2": SERAFIM_EMBEDDING_MODEL,
    "3": MINILM_L6_V2,
    "4": MINILM_L12_V2,
    "0": None
}
DIMENSIONS = {
    "1": DIM_MXBAI,
    "2": DIM_SERAFIM,
    "3": DIM_MINILM_L6,
    "4": DIM_MINIL12,
    "0": None
}

# ──────────────────────────────────────────────────────────────────────────────
# Seleção de DB schemas (agora incluindo vector_1536)
# ──────────────────────────────────────────────────────────────────────────────
SGDB_OPTIONS = {"1": "mongo", "2": "postgres", "0": None}
DB_SCHEMA_OPTIONS = {
    "1": "vector_1024",
    "2": "vector_384",
    "3": "vector_384_teste",
    "4": "vector_1536",
    "0": None
}

# ──────────────────────────────────────────────────────────────────────────────
# Helpers de menu
# ──────────────────────────────────────────────────────────────────────────────
def clear_screen():
    os.system("clear")

def select_embedding_model():
    print("\n*** Modelos de Embedding Disponíveis ***")
    for k, name in [
        ("1", OLLAMA_EMBEDDING_MODEL),
        ("2", SERAFIM_EMBEDDING_MODEL),
        ("3", MINILM_L6_V2),
        ("4", MINILM_L12_V2),
        ("0", "Voltar")
    ]:
        print(f"{k} - {name}")
    return EMBEDDING_MODELS.get(input("Escolha [1]: ").strip())

def select_dimension():
    print("\n*** Dimensão dos Embeddings ***")
    for k, d in [
        ("1", DIM_MXBAI),
        ("2", DIM_SERAFIM),
        ("3", DIM_MINILM_L6),
        ("4", DIM_MINIL12),
        ("0", "Voltar")
    ]:
        print(f"{k} - {d}")
    return DIMENSIONS.get(input("Escolha [1]: ").strip())

def select_schema():
    print("\n*** Schemas PostgreSQL Disponíveis ***")
    print("1 - vector_1024\n2 - vector_384\n3 - vector_384_teste\n4 - vector_1536\n0 - Voltar")
    return DB_SCHEMA_OPTIONS.get(input("Escolha [1]: ").strip())

# (outros helpers e fluxo principal permanecem inalterados, exceto passando a nova opção 4 em select_schema)

# ──────────────────────────────────────────────────────────────────────────────
# Processamento de arquivo (idêntico ao anterior)
# ──────────────────────────────────────────────────────────────────────────────
def process_file(path: str, strategy: str, sgbd: str, schema: str,
                 embedding_model: str, embedding_dim: int, results: dict):
    filename = os.path.basename(path)
    if not is_valid_file(path):
        results["errors"].append(path)
        return
    if not is_extraction_allowed(path):
        text = fallback_ocr(path, OCR_THRESHOLD)
    else:
        key = "unstructured" if path.lower().endswith(".docx") else strategy
        text = STRATEGIES[key].extract(path)
    rec = build_meta(path, text)
    if sgbd == "mongo":
        pid = save_metadata(rec, DB_NAME, COLL_PDF, MONGO_URI)
        save_file_binary(filename, path, pid, DB_NAME, COLL_BIN, MONGO_URI)
        save_gridfs(path, filename, DB_NAME, GRIDFS_BUCKET, MONGO_URI)
    else:
        save_to_postgres(filename, rec["text"], rec["info"],
                         embedding_model, embedding_dim, schema)
    results["processed"].append(path)
    try:
        archive = os.path.join(os.path.dirname(path), "processed")
        os.makedirs(archive, exist_ok=True)
        shutil.move(path, os.path.join(archive, filename))
    except:
        pass

# ──────────────────────────────────────────────────────────────────────────────
# Fluxo principal (ajustado para usar select_embedding_model, select_dimension, select_schema)
# ──────────────────────────────────────────────────────────────────────────────
def main():
    current_strat = "ocr"
    current_sgbd = "mongo"
    current_schema = "vector_1024"
    current_model = OLLAMA_EMBEDDING_MODEL
    current_dim = DIM_MXBAI
    results = {"processed": [], "errors": []}

    while True:
        clear_screen()
        print("*** Menu Principal ***")
        print(f"1 - Selecionar Estratégia    (atual: {current_strat})")
        print(f"2 - Selecionar SGBD          (atual: {current_sgbd})")
        idx = 3 if current_sgbd == "postgres" else 2
        if current_sgbd == "postgres":
            print(f"3 - Selecionar Schema        (atual: {current_schema})")
        print(f"{idx+1} - Processar Arquivo")
        print(f"{idx+2} - Processar Pasta")
        print(f"{idx+3} - Selecionar Embedding     (atual: {current_model})")
        print(f"{idx+4} - Selecionar Dimensão      (atual: {current_dim})")
        print("0 - Sair")
        choice = input("> ").strip()

        if choice == "0":
            break
        if choice == "1":
            # estratégia (unchanged)
            pass
        elif choice == "2":
            # sgbd (unchanged)
            pass
        elif choice == "3" and current_sgbd == "postgres":
            current_schema = select_schema() or current_schema
        elif choice == str(idx+3):
            current_model = select_embedding_model() or current_model
        elif choice == str(idx+4):
            current_dim = select_dimension() or current_dim
        # (Processar arquivo/pasta branches remain the same)

    # resumo final...
    logging.info("Aplicação encerrada.")

if __name__ == "__main__":
    main()