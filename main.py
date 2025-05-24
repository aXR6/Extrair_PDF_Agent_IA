# main.py (CLI de processamento)
#!/usr/bin/env python3
import os
import logging
from tqdm import tqdm
import shutil

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
STRAT_OPTIONS = {"1": "pypdf", "2": "pdfminer", "3": "pdfminer-low",
                 "4": "unstructured", "5": "ocr", "6": "plumber",
                 "7": "tika", "8": "pymupdf4llm", "0": None}

# ──────────────────────────────────────────────────────────────────────────────
# Modelos e dimensões de embeddings
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
# Seleção de SGBD e Schemas PostgreSQL
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

def select_strategy():
    print("\n*** Estratégias Disponíveis ***")
    for k, label in [
        ("1","PyPDFLoader"),("2","PDFMinerLoader"),("3","PDFMiner Low-Level"),
        ("4","Unstructured (.docx)"),("5","OCR"),("6","PDFPlumber"),
        ("7","Apache Tika"),("8","PyMuPDF4LLM"),("0","Voltar")
    ]:
        print(f"{k} - {label}")
    return STRAT_OPTIONS.get(input("Escolha [5]: ").strip())

def select_sgbd():
    print("\n*** Seleção de SGBD ***")
    print("1 - MongoDB\n2 - PostgreSQL\n0 - Voltar")
    return SGDB_OPTIONS.get(input("Escolha [1]: ").strip())

def select_schema():
    print("\n*** Schemas PostgreSQL Disponíveis ***")
    print("1 - vector_1024\n2 - vector_384\n3 - vector_384_teste\n4 - vector_1536\n0 - Voltar")
    return DB_SCHEMA_OPTIONS.get(input("Escolha [1]: ").strip())

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

# ──────────────────────────────────────────────────────────────────────────────
# Processamento de arquivo
# ──────────────────────────────────────────────────────────────────────────────
def process_file(path, strategy, sgbd, schema, embedding_model, embedding_dim, results):
    filename = os.path.basename(path)
    if not is_valid_file(path):
        results["errors"].append(path); return
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
    # arquivar
    try:
        dst = os.path.join(os.path.dirname(path), "processed")
        os.makedirs(dst, exist_ok=True)
        shutil.move(path, os.path.join(dst, filename))
    except:
        pass

# ──────────────────────────────────────────────────────────────────────────────
# Fluxo principal
# ──────────────────────────────────────────────────────────────────────────────
def main():
    current_strat   = "ocr"
    current_sgbd    = "mongo"
    current_schema  = "vector_1024"
    current_model   = OLLAMA_EMBEDDING_MODEL
    current_dim     = DIM_MXBAI
    results = {"processed": [], "errors": []}

    while True:
        clear_screen()
        print("*** Menu Principal ***")
        print(f"1 - Selecionar Estratégia    (atual: {current_strat})")
        print(f"2 - Selecionar SGBD          (atual: {current_sgbd})")
        offset = 1 if current_sgbd == "postgres" else 0
        if offset:
            print(f"3 - Selecionar Schema        (atual: {current_schema})")
        print(f"{3+offset} - Processar Arquivo")
        print(f"{4+offset} - Processar Pasta")
        print(f"{5+offset} - Selecionar Embedding     (atual: {current_model})")
        print(f"{6+offset} - Selecionar Dimensão      (atual: {current_dim})")
        print("0 - Sair")
        choice = input("> ").strip()

        if choice == "0":
            break
        elif choice == "1":
            sel = select_strategy()
            if sel: current_strat = sel
        elif choice == "2":
            sel = select_sgbd()
            if sel: current_sgbd = sel
        elif choice == "3" and current_sgbd == "postgres":
            sel = select_schema()
            if sel: current_schema = sel
        elif choice == str(3+offset):
            p = input("Caminho do arquivo: ").strip()
            process_file(p, current_strat, current_sgbd,
                         current_schema, current_model, current_dim, results)
            input("\nENTER para voltar…")
        elif choice == str(4+offset):
            folder = input("Caminho da pasta: ").strip()
            # coleta recursiva...
            all_files = []
            parent = os.path.dirname(folder)
            roots = [folder] + ([os.path.join(parent,d) for d in os.listdir(parent)
                                 if os.path.isdir(os.path.join(parent,d)) and os.path.join(parent,d)!=folder]
                               if os.path.isdir(parent) else [])
            for root in roots:
                for dp, dns, fns in os.walk(root):
                    if "processed" in dns: dns.remove("processed")
                    for fn in fns:
                        if fn.lower().endswith((".pdf",".docx")):
                            all_files.append(os.path.join(dp,fn))
            if not all_files:
                input("Nenhum PDF/DOCX encontrado. ENTER…"); continue

            print(f"Processando {len(all_files)} arquivos…")
            for path in tqdm(all_files, desc="Arquivos", unit="file"):
                process_file(path, current_strat, current_sgbd,
                             current_schema, current_model, current_dim, results)
            input("\nENTER para voltar…")
        elif choice == str(5+offset):
            sel = select_embedding_model()
            if sel: current_model = sel
        elif choice == str(6+offset):
            sel = select_dimension()
            if sel: current_dim = sel
        else:
            input("Opção inválida. ENTER para tentar novamente…")

    clear_screen()
    print("\n=== Resumo Final ===")
    print(f"Processados: {len(results['processed'])}")
    if results["errors"]:
        print(f"Erros ({len(results['errors'])}):")
        for e in results["errors"]:
            print(f"  - {e}")
    print("\nEncerrando.")
    logging.info("Aplicação encerrada.")

if __name__ == "__main__":
    main()