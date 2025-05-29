# main.py
#!/usr/bin/env python3
import os
import sys
import argparse
import time
import logging
from tqdm import tqdm

# garante imports locais
sys.path.insert(0, os.path.dirname(__file__))

from config import (
    OLLAMA_EMBEDDING_MODEL, SERAFIM_EMBEDDING_MODEL,
    MINILM_L6_V2, MINILM_L12_V2, MPNET_EMBEDDING_MODEL,
    DIM_MXBAI, DIM_SERAFIM, DIM_MINILM_L6, DIM_MINIL12, DIM_MPNET,
    OCR_THRESHOLD, validate_config
)
from extractors import extract_text
from utils import setup_logging, is_valid_file, build_record, repair_pdf
from pg_storage import save_to_postgres
from adaptive_chunker import get_sbert_model

# valida e inicializa SBERT + logs
validate_config()
setup_logging()
get_sbert_model()

# opções de menu
STRATEGY_OPTIONS = [
    "pypdf", "pdfminer", "pdfminer-low", "unstructured",
    "ocr", "plumber", "tika", "pymupdf4llm"
]
EMBED_MODELS = {
    "1": OLLAMA_EMBEDDING_MODEL,
    "2": SERAFIM_EMBEDDING_MODEL,
    "3": MINILM_L6_V2,
    "4": MINILM_L12_V2,
    "5": MPNET_EMBEDDING_MODEL
}
DIMENSIONS = {
    "1": DIM_MXBAI,
    "2": DIM_SERAFIM,
    "3": DIM_MINILM_L6,
    "4": DIM_MINIL12,
    "5": DIM_MPNET
}

def clear_screen():
    os.system("clear")


def select_strategy():
    print("\n*** Selecione Estratégia ***")
    for i, k in enumerate(STRATEGY_OPTIONS, 1):
        print(f"{i} - {k}")
    c = input("Escolha [ocr]: ").strip()
    return STRATEGY_OPTIONS[int(c)-1] if c.isdigit() and 1 <= int(c) <= len(STRATEGY_OPTIONS) else "ocr"


def select_embedding():
    print("\n*** Selecione Embedding ***")
    for k, n in EMBED_MODELS.items():
        print(f"{k} - {n}")
    return EMBED_MODELS.get(input("Escolha [1]: ").strip(), OLLAMA_EMBEDDING_MODEL)


def select_dimension():
    print("\n*** Selecione Dimensão ***")
    for k, d in DIMENSIONS.items():
        print(f"{k} - {d}")
    return DIMENSIONS.get(input("Escolha [1]: ").strip(), DIM_MXBAI)


def process_file(path, strat, model, dim, stats):
    p = os.path.normpath(path.strip())
    base, ext = os.path.splitext(p)
    p2 = base.rstrip() + ext
    if p2 != p:
        try:
            os.rename(p, p2)
        except Exception:
            pass

    if not is_valid_file(p2):
        stats['errors'] += 1
        return

    text = extract_text(p2, strat)
    if not text or len(text.strip()) < OCR_THRESHOLD:
        logging.error(f"Não foi possível extrair: {os.path.basename(p2)}")
        stats['errors'] += 1
        return

    rec = build_record(p2, text)
    try:
        save_to_postgres(
            os.path.basename(p2), rec['text'], rec['info'],
            model, dim
        )
        stats['processed'] += 1
    except Exception as e:
        logging.error(f"Erro salvando {p2}: {e}")
        stats['errors'] += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    strat = "ocr"
    model = OLLAMA_EMBEDDING_MODEL
    dim = DIM_MXBAI
    stats = {"processed": 0, "errors": 0}

    while True:
        clear_screen()
        print("*** Menu Principal ***")
        print(f"1 - Estratégia (atual: {strat})")
        print(f"2 - Embedding  (atual: {model})")
        print(f"3 - Dimensão   (atual: {dim})")
        print("4 - Arquivo")
        print("5 - Pasta")
        print("0 - Sair")
        c = input("> ").strip()

        if c == "0":
            break
        elif c == "1":
            strat = select_strategy()
        elif c == "2":
            model = select_embedding()
        elif c == "3":
            dim = select_dimension()
        elif c == "4":
            f = input("Arquivo: ").strip()
            start = time.perf_counter()
            process_file(f, strat, model, dim, stats)
            dt = time.perf_counter() - start
            print(f"→ {dt:.2f}s • P: {stats['processed']} • E: {stats['errors']}")
            input("ENTER…")
        elif c == "5":
            d = input("Pasta: ").strip()
            files = [
                os.path.join(root, fname)
                for root, _, files_ in os.walk(d)
                for fname in files_
                if fname.lower().endswith((".pdf", ".docx", ".png", ".jpg", ".jpeg", ".tiff"))
            ]
            print(f"Total: {len(files)}")
            start = time.perf_counter()
            pbar = tqdm(files, unit="arquivo")
            for path in pbar:
                process_file(path, strat, model, dim, stats)
                pbar.set_postfix(P=stats['processed'], E=stats['errors'])
            pbar.close()
            dt = time.perf_counter() - start
            print(f"=== Resumo ===\n P: {stats['processed']}\n E: {stats['errors']}\n T: {dt:.2f}s")
            input("ENTER…")
        else:
            input("Inválido…")

    clear_screen()
    print(f"Processados: {stats['processed']}  Erros: {stats['errors']}")


if __name__ == "__main__":
    main()