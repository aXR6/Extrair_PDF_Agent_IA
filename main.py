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
    PG_SCHEMA_DEFAULT, PG_SCHEMAS,
    OLLAMA_EMBEDDING_MODEL, SERAFIM_EMBEDDING_MODEL,
    MINILM_L6_V2, MINILM_L12_V2, MPNET_EMBEDDING_MODEL,
    DIM_MXBAI, DIM_SERAFIM, DIM_MINILM_L6, DIM_MINIL12, DIM_MPNET,
    OCR_THRESHOLD, validate_config
)
from extractors import extract_text
from utils import setup_logging, is_valid_file, build_record, repair_pdf
from pg_storage import save_to_postgres
from adaptive_chunker import get_sbert_model

# valida e inicializa
validate_config()
setup_logging()
get_sbert_model()

# opções de menus
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

def select_schema():
    print("\\n*** Selecione schema ***")
    for i,s in enumerate(PG_SCHEMAS,1):
        mark = " (default)" if s==PG_SCHEMA_DEFAULT else ""
        print(f"{i} - {s}{mark}")
    c = input("Escolha [default]: ").strip()
    return PG_SCHEMAS[int(c)-1] if c.isdigit() and 1<=int(c)<=len(PG_SCHEMAS) else PG_SCHEMA_DEFAULT

def select_strategy():
    print("\\n*** Selecione estratégia ***")
    for i,k in enumerate(STRATEGY_OPTIONS,1):
        print(f"{i} - {k}")
    c = input("Escolha [ocr]: ").strip()
    return STRATEGY_OPTIONS[int(c)-1] if c.isdigit() and 1<=int(c)<=len(STRATEGY_OPTIONS) else "ocr"

def select_embedding():
    print("\\n*** Selecione modelo de embedding ***")
    for k,n in EMBED_MODELS.items():
        print(f"{k} - {n}")
    return EMBED_MODELS.get(input("Escolha [1]: ").strip(), OLLAMA_EMBEDDING_MODEL)

def select_dimension():
    print("\\n*** Selecione dimensão ***")
    for k,d in DIMENSIONS.items():
        print(f"{k} - {d}")
    return DIMENSIONS.get(input("Escolha [1]: ").strip(), DIM_MXBAI)

def process_file(path, strat, schema, model, dim, stats):
    # normaliza nome
    p = os.path.normpath(path.strip())
    base, ext = os.path.splitext(p)
    p2 = base.rstrip() + ext
    if p2!=p:
        try: os.rename(p,p2)
        except: pass
    p = p2

    if not is_valid_file(p):
        stats['errors']+=1
        return

    # reparo + extração
    text = extract_text(p, strat)
    if not text or len(text.strip())<OCR_THRESHOLD:
        logging.error(f"Não extraído: {os.path.basename(p)}")
        stats['errors']+=1
        return

    rec = build_record(p, text)
    try:
        save_to_postgres(os.path.basename(p), rec['text'], rec['info'], model, dim, schema)
        stats['processed']+=1
    except Exception as e:
        logging.error(f"Erro ao salvar {p}: {e}")
        stats['errors']+=1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    schema = PG_SCHEMA_DEFAULT
    strat  = "ocr"
    model  = OLLAMA_EMBEDDING_MODEL
    dim    = DIM_MXBAI
    stats  = {"processed":0,"errors":0}

    while True:
        clear_screen()
        print("*** Menu Principal ***")
        print(f"1 - Schema         (atual: {schema})")
        print(f"2 - Estratégia     (atual: {strat})")
        print(f"3 - Embedding      (atual: {model})")
        print(f"4 - Dimensão       (atual: {dim})")
        print("5 - Processar Arquivo")
        print("6 - Processar Pasta")
        print("0 - Sair")
        c = input("> ").strip()

        if c=="0":
            break
        elif c=="1":
            schema = select_schema()
        elif c=="2":
            strat = select_strategy()
        elif c=="3":
            model = select_embedding()
        elif c=="4":
            dim = select_dimension()
        elif c=="5":
            f = input("Arquivo: ").strip()
            start = time.perf_counter()
            process_file(f, strat, schema, model, dim, stats)
            dt = time.perf_counter()-start
            print(f"→ {dt:.2f}s  Processados: {stats['processed']}  Erros: {stats['errors']}")
            input("ENTER…")
        elif c=="6":
            d = input("Pasta: ").strip()
            files = [os.path.join(r,f) for r,_,fs in os.walk(d) for f in fs if f.lower().endswith((".pdf",".docx"))]
            print(f"Total: {len(files)}")
            start = time.perf_counter()
            pbar = tqdm(files, unit="arquivo")
            for path in pbar:
                process_file(path, strat, schema, model, dim, stats)
                pbar.set_postfix(proc=stats['processed'], err=stats['errors'])
            pbar.close()
            dt = time.perf_counter()-start
            print(f"=== Resumo ===\nProcessados: {stats['processed']}\nErros: {stats['errors']}\nTempo: {dt:.2f}s")
            input("ENTER…")
        else:
            input("Opção inválida…")

    clear_screen()
    print(f"Processados: {stats['processed']}\nErros: {stats['errors']}")

if __name__=="__main__":
    main()
