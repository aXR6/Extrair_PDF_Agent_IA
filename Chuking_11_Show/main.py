#main.py
#!/usr/bin/env python3
import os
import logging
from tqdm import tqdm

from config import (
    MONGO_URI,
    DB_NAME,
    COLL_PDF,
    COLL_BIN,
    GRIDFS_BUCKET,
    OCR_THRESHOLD
)
from utils import (
    setup_logging,
    is_valid_file,
    build_record as build_meta
)
from extractors import (
    is_extraction_allowed,
    fallback_ocr,
    PyPDFStrategy,
    PDFMinerStrategy,
    PDFMinerLowLevelStrategy,
    UnstructuredStrategy,
    OCRStrategy,
    PDFPlumberStrategy,
    TikaStrategy,
    PyMuPDF4LLMStrategy
)
from storage import save_metadata, save_file_binary, save_gridfs
from pg_storage import save_to_postgres

# ──────────────────────────────────────────────────────────────────────────────
# Inicialização de logs
# ──────────────────────────────────────────────────────────────────────────────
logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("PyPDF2").setLevel(logging.ERROR)
setup_logging()

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
STRAT_OPTIONS = {
    "1": "pypdf", "2": "pdfminer", "3": "pdfminer-low",
    "4": "unstructured", "5": "ocr", "6": "plumber",
    "7": "tika", "8": "pymupdf4llm", "0": None
}

# ──────────────────────────────────────────────────────────────────────────────
# Seleção de SGBD e Schemas PostgreSQL
# ──────────────────────────────────────────────────────────────────────────────
SGDB_OPTIONS = {
    "1": "mongo",
    "2": "postgres",
    "0": None
}
DB_SCHEMA_OPTIONS = {
    "1": "vector_1024",
    "2": "vector_384",
    "3": "vector_384_teste",
    "0": None
}

# ──────────────────────────────────────────────────────────────────────────────
# Modelos e dimensões de embeddings
# ──────────────────────────────────────────────────────────────────────────────
EMBEDDING_MODELS = {
    "1": "mxbai-embed-large",
    "2": "jvanhoof/all-MiniLM-L6-multilingual-v2-en-es-pt-pt-br-v2",
    "3": "sentence-transformers/all-MiniLM-L6-v2",
    "4": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "0": None
}
DIMENSIONS = {
    "1": 1024,
    "2": 384,
    "0": None
}

# ──────────────────────────────────────────────────────────────────────────────
# Helpers de menu
# ──────────────────────────────────────────────────────────────────────────────
def clear_screen():
    os.system("clear")  # 'cls' no Windows


def select_strategy():
    print("\n*** Estratégias Disponíveis ***")
    for k, label in [
        ("1","PyPDFLoader"),
        ("2","PDFMinerLoader"),
        ("3","PDFMiner Low-Level"),
        ("4","Unstructured (.docx)"),
        ("5","OCR"),
        ("6","PDFPlumber"),
        ("7","Apache Tika"),
        ("8","PyMuPDF4LLM"),
        ("0","Voltar")
    ]:
        print(f"{k} - {label}")
    return STRAT_OPTIONS.get(input("Escolha [5]: ").strip())


def select_sgbd():
    print("\n*** Seleção de SGBD ***")
    print("1 - MongoDB")
    print("2 - PostgreSQL")
    print("0 - Voltar")
    return SGDB_OPTIONS.get(input("Escolha [1]: ").strip())


def select_schema():
    print("\n*** Schemas PostgreSQL Disponíveis ***")
    print("1 - vector_1024")
    print("2 - vector_384")
    print("3 - vector_384_teste")
    print("0 - Voltar")
    return DB_SCHEMA_OPTIONS.get(input("Escolha [1]: ").strip())


def select_embedding_model():
    print("\n*** Modelos de Embeddings Disponíveis ***")
    for k, label in [
        ("1","mxbai-embed-large (Ollama)"),
        ("2","Multilingual MiniLM-L6-v2"),
        ("3","all-MiniLM-L6-v2"),
        ("4","paraphrase-multilingual-MiniLM-L12-v2"),
        ("0","Voltar")
    ]:
        print(f"{k} - {label}")
    return EMBEDDING_MODELS.get(input("Escolha [1]: ").strip())


def select_dimension():
    print("\n*** Dimensão dos Embeddings ***")
    print("1 - 1024 (padrão mxbai-embed-large)")
    print("2 - 384  (MiniLM-L6)")
    print("0 - Voltar")
    return DIMENSIONS.get(input("Escolha [1]: ").strip())

# ──────────────────────────────────────────────────────────────────────────────
# Processamento de arquivo
# ──────────────────────────────────────────────────────────────────────────────
def process_file(
    path: str,
    strategy: str,
    sgbd: str,
    schema: str,
    embedding_model: str,
    embedding_dim: int,
    results: dict
):
    filename = os.path.basename(path)
    print(f"\n→ Processando: {filename} [estratégia={strategy}, sgbd={sgbd}]")

    if not is_valid_file(path):
        print("  Arquivo inválido")
        results["errors"].append(path)
        return

    if not is_extraction_allowed(path):
        print("  Usando OCR fallback…")
        text = fallback_ocr(path, OCR_THRESHOLD)
    else:
        key = "unstructured" if path.lower().endswith(".docx") else strategy
        print(f"  Extraindo com: {key}")
        text = STRATEGIES[key].extract(path)

    rec = build_meta(path, text)

    if sgbd == "mongo":
        print("  Salvando no MongoDB…")
        pid = save_metadata(rec, DB_NAME, COLL_PDF, MONGO_URI)
        save_file_binary(filename, path, pid, DB_NAME, COLL_BIN, MONGO_URI)
        save_gridfs(path, filename, DB_NAME, GRIDFS_BUCKET, MONGO_URI)
    else:
        print(f"  Salvando no PostgreSQL ({schema})…")
        save_to_postgres(
            filename,
            rec["text"],
            rec["info"],
            embedding_model,
            embedding_dim,
            schema
        )

    results["processed"].append(path)
    print(f"→ {filename} concluído")

# ──────────────────────────────────────────────────────────────────────────────
# Fluxo principal
# ──────────────────────────────────────────────────────────────────────────────

def main():
    current_strat   = "ocr"
    current_sgbd    = "mongo"
    current_schema  = "vector_1024"
    current_model   = "mxbai-embed-large"
    current_dim     = 1024
    results = {"processed": [], "errors": []}

    while True:
        clear_screen()
        print("*** Menu Principal ***")
        print(f"1 - Selecionar Estratégia    (atual: {current_strat})")
        print(f"2 - Selecionar SGBD          (atual: {current_sgbd})")
        if current_sgbd == "postgres":
            print(f"3 - Selecionar Schema        (atual: {current_schema})")
            offset = 1
        else:
            offset = 0
        print(f"{3+offset} - Processar Arquivo")
        print(f"{4+offset} - Processar Pasta")
        print(f"{5+offset} - Selecionar Embedding     (atual: {current_model})")
        print(f"{6+offset} - Selecionar Dimensão      (atual: {current_dim})")
        print("0 - Sair")
        choice = input("> ").strip()

        if choice == "0":
            break
        elif choice == "1":
            clear_screen()
            sel = select_strategy()
            if sel:
                current_strat = sel
        elif choice == "2":
            clear_screen()
            sel = select_sgbd()
            if sel:
                current_sgbd = sel
        elif choice == "3" and current_sgbd == "postgres":
            clear_screen()
            sel = select_schema()
            if sel:
                current_schema = sel
        elif choice == str(3+offset):
            clear_screen()
            p = input("Caminho do arquivo: ").strip()
            process_file(p, current_strat, current_sgbd, current_schema, current_model, current_dim, results)
            input("\nENTER para voltar…")
        elif choice == str(4+offset):
            clear_screen()
            folder = input("Caminho da pasta: ").strip()
            files = [f for f in os.listdir(folder) if f.lower().endswith((".pdf", ".docx"))]
            if not files:
                print("Nenhum PDF/DOCX encontrado.")
                input("\nENTER para voltar…")
                continue
            print(f"Processando {len(files)} arquivos…")
            for f in tqdm(files, desc="Arquivos", unit="file"):
                process_file(
                    os.path.join(folder, f),
                    current_strat,
                    current_sgbd,
                    current_schema,
                    current_model,
                    current_dim,
                    results
                )
            input("\nENTER para voltar…")
        elif choice == str(5+offset):
            clear_screen()
            sel = select_embedding_model()
            if sel:
                current_model = sel
        elif choice == str(6+offset):
            clear_screen()
            sel = select_dimension()
            if sel:
                current_dim = sel
        else:
            print("Opção inválida.")
            input("\nENTER para tentar novamente…")

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