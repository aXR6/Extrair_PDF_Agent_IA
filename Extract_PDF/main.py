#!/usr/bin/env python3
import os
import logging
from tqdm import tqdm

from config import (
    MONGO_URI, DB_NAME, COLL_PDF, COLL_BIN, GRIDFS_BUCKET, OCR_THRESHOLD
)
from utils import setup_logging, is_valid_file, generate_report, build_record as build_meta
from extractors import (
    is_extraction_allowed,
    fallback_ocr,
    build_record,
    PyPDFStrategy,
    PDFMinerStrategy,
    PDFMinerLowLevelStrategy,
    UnstructuredStrategy,
    OCRStrategy,
    PDFPlumberStrategy,
    TikaStrategy,
    PyMuPDF4LLMStrategy  # Nova estratégia importada
)
from storage import save_metadata, save_file_binary, save_gridfs
from pg_storage import save_to_postgres

# Suprimir warnings irrelevantes
logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("PyPDF2").setLevel(logging.ERROR)
setup_logging()

# Define estratégias e opções de DB
STRATEGIES = {
    "pypdf":        PyPDFStrategy(),
    "pdfminer":     PDFMinerStrategy(),
    "pdfminer-low": PDFMinerLowLevelStrategy(),
    "unstructured": UnstructuredStrategy(),
    "ocr":          OCRStrategy(threshold=OCR_THRESHOLD),
    "plumber":      PDFPlumberStrategy(),
    "tika":         TikaStrategy(),
    "pymupdf4llm":  PyMuPDF4LLMStrategy(),  # Adicionada aqui
}
DB_OPTIONS = {"1": "mongo", "2": "postgres", "0": None}
STRAT_OPTIONS = {
    "1": "pypdf",
    "2": "pdfminer",
    "3": "pdfminer-low",
    "4": "unstructured",
    "5": "ocr",
    "6": "plumber",
    "7": "tika",
    "8": "pymupdf4llm",  # Mapeamento da nova opção
    "0": None
}

def select_strategy():
    print("\n*** Estratégias Disponíveis ***")
    for key, name in [
        ("1", "PyPDFLoader (LangChain)"),
        ("2", "PDFMinerLoader (LangChain)"),
        ("3", "PDFMiner Low-Level (pdfminer.six)"),
        ("4", "Unstructured (.docx)"),
        ("5", "OCR"),
        ("6", "PDFPlumber"),
        ("7", "Apache Tika"),
        ("8", "PyMuPDF4LLM (Markdown)"),  # Nova entrada no menu
        ("0", "Voltar")
    ]:
        print(f"{key} - {name}")
    choice = input("Escolha [5]: ").strip()
    return STRAT_OPTIONS.get(choice)

def select_database():
    print("\n*** Bancos de Dados ***")
    print("1 - MongoDB")
    print("2 - PostgreSQL")
    print("0 - Voltar")
    choice = input("Escolha [1]: ").strip()
    return DB_OPTIONS.get(choice)

def process_file(path, strategy, db_choice, results):
    print(f"\nIniciando processamento: {os.path.basename(path)} usando {strategy} no DB {db_choice}")
    if not is_valid_file(path):
        print(f"Arquivo inválido: {path}")
        results["errors"].append(path)
        return

    if not is_extraction_allowed(path):
        print("Extração direta não permitida. Aplicando OCR fallback...")
        text = fallback_ocr(path, OCR_THRESHOLD)
    else:
        ext = os.path.splitext(path)[1].lower()
        key = "unstructured" if ext == ".docx" else strategy
        print(f"Extraindo com estratégia: {key}")
        text = STRATEGIES.get(key, STRATEGIES["ocr"]).extract(path)

    rec = build_meta(path, text)
    print("Salvando dados no banco...")
    if db_choice == "mongo":
        pid = save_metadata(rec, DB_NAME, COLL_PDF, MONGO_URI)
        save_file_binary(os.path.basename(path), path, pid,
                         DB_NAME, COLL_BIN, MONGO_URI)
        save_gridfs(path, os.path.basename(path),
                    DB_NAME, GRIDFS_BUCKET, MONGO_URI)
    elif db_choice == "postgres":
        save_to_postgres(os.path.basename(path), rec["text"], rec["info"])

    results["processed"].append(path)
    print(f"Concluído: {os.path.basename(path)}")

def main():
    current_db = "mongo"
    current_strat = "ocr"
    results = {"processed": [], "errors": []}

    while True:
        print("\n*** Menu Principal ***")
        print(f"1 - Selecionar Estratégia (atual: {current_strat})")
        print(f"2 - Selecionar Banco (atual: {current_db})")
        print("3 - Processar Arquivo")
        print("4 - Processar Pasta")
        print("0 - Sair")
        choice = input("> ").strip()

        if choice == "0":
            break
        elif choice == "1":
            new_strat = select_strategy()
            if new_strat is not None:
                current_strat = new_strat
            else:
                print("Retornando ao menu principal.")
        elif choice == "2":
            new_db = select_database()
            if new_db is not None:
                current_db = new_db
            else:
                print("Retornando ao menu principal.")
        elif choice == "3":
            path = input("Digite o caminho do arquivo: ").strip()
            process_file(path, current_strat, current_db, results)
        elif choice == "4":
            folder = input("Digite o caminho da pasta: ").strip()
            files = [f for f in os.listdir(folder) if f.lower().endswith((".pdf", ".docx"))]
            if not files:
                print("Nenhum arquivo PDF/DOCX encontrado na pasta.")
                continue
            print(f"Iniciando processamento de {len(files)} arquivos...")
            for fname in tqdm(files, desc="Processando arquivos", unit="file"):
                full = os.path.join(folder, fname)
                process_file(full, current_strat, current_db, results)
        else:
            print("Opção inválida.")

    print("\nResumo do processamento:")
    print(f"Processados: {len(results['processed'])}")
    if results['errors']:
        print(f"Erros em: {len(results['errors'])} arquivos")
        for e in results['errors']:
            print(f" - {e}")
    print("Fim.")
    logging.info("Encerrando.")

if __name__ == "__main__":
    main()