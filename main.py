#!/usr/bin/env python3
import os
import sys
import argparse
import time
import logging
from tqdm import tqdm

# Garante que o diretório do script esteja no path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    PG_HOST, PG_PORT, PG_USER, PG_PASSWORD,
    PG_SCHEMAS, PG_SCHEMA_DEFAULT,
    OLLAMA_EMBEDDING_MODEL, SERAFIM_EMBEDDING_MODEL,
    MINILM_L6_V2, MINILM_L12_V2, MPNET_EMBEDDING_MODEL,
    DIM_MXBAI, DIM_SERAFIM, DIM_MINILM_L6, DIM_MINIL12, DIM_MPNET,
    SBERT_MODEL_NAME, OCR_THRESHOLD,
    validate_config
)
from adaptive_chunker import get_sbert_model
from extractors import extract_text, STRATEGIES_MAP
from pg_storage import save_to_postgres
from utils import setup_logging, is_valid_file, build_record, repair_pdf

# Valida env, inicializa logs e pré-carrega SBERT
validate_config()
setup_logging()
get_sbert_model(SBERT_MODEL_NAME)

# Mapeamento de modelos e dimensões de embedding
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

# Helpers
def clear_screen(): os.system("clear")


def process_file(path, strategy, schema, model, dim, results, verbose=False):
    """
    Processa um arquivo PDF/DOCX:
      1) Limpa e normaliza path (remove espaços antes da extensão)
      2) Repara PDF corrompido (pikepdf + Ghostscript)
      3) Extrai texto (pipeline unificado)
      4) Persiste no PostgreSQL
    """
    # 1) Clean e normaliza path
    original = path
    path = original.strip()
    path = os.path.normpath(path)
    base, ext = os.path.splitext(path)
    path = base.rstrip() + ext  # remove espaços antes da extensão

    # 2) Tenta reparar o PDF
    path = repair_pdf(path)
    fn = os.path.basename(path)

    # 3) Valida arquivo
    if not is_valid_file(path):
        results['errors'] += 1
        return False

    # 4) Extrai texto via pipeline unificado
    text = extract_text(path, strategy)
    if not text or len(text.strip()) < OCR_THRESHOLD:
        logging.error(f"Não foi possível extrair texto de {fn}. Pulando.")
        results['errors'] += 1
        return False

    # 5) Persistência
    rec = build_record(path, text)
    try:
        save_to_postgres(fn, rec['text'], rec['info'], model, dim, schema)
        results['processed'] += 1
        return True
    except Exception as e:
        logging.error(f"Erro salvando {fn}: {e}")
        results['errors'] += 1
        return False


def select_schema():
    print("\n*** Selecione Schema PostgreSQL ***")
    for idx, sch in enumerate(PG_SCHEMAS, start=1):
        default = " (default)" if sch == PG_SCHEMA_DEFAULT else ""
        print(f"{idx} - {sch}{default}")
    choice = input("Escolha [default]: ").strip()
    return PG_SCHEMAS[int(choice)-1] if choice.isdigit() and 1 <= int(choice) <= len(PG_SCHEMAS) else PG_SCHEMA_DEFAULT


def select_strategy():
    print("\n*** Selecione Estratégia de Extração ***")
    opts = list(STRATEGIES_MAP.keys())
    for idx, key in enumerate(opts, start=1):
        print(f"{idx} - {key}")
    choice = input("Escolha [ocr]: ").strip()
    return opts[int(choice)-1] if choice.isdigit() and 1 <= int(choice) <= len(opts) else 'ocr'


def select_embedding():
    print("\n*** Selecione Modelo de Embedding ***")
    for k, name in EMBED_MODELS.items():
        print(f"{k} - {name}")
    choice = input("Escolha [1]: ").strip()
    return EMBED_MODELS.get(choice, OLLAMA_EMBEDDING_MODEL)


def select_dimension():
    print("\n*** Selecione Dimensão de Embedding ***")
    for k, d in DIMENSIONS.items():
        print(f"{k} - {d}")
    choice = input("Escolha [1]: ").strip()
    return DIMENSIONS.get(choice, DIM_MXBAI)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', action='store_true', help='Imprime logs detalhados')
    args = parser.parse_args()

    schema = PG_SCHEMA_DEFAULT
    strategy = 'ocr'
    model = OLLAMA_EMBEDDING_MODEL
    dim = DIM_MXBAI
    results = {'processed': 0, 'errors': 0}

    while True:
        clear_screen()
        print("*** Menu Principal ***")
        print(f"1 - Schema         (atual: {schema})")
        print(f"2 - Estratégia     (atual: {strategy})")
        print(f"3 - Embedding Model(atual: {model})")
        print(f"4 - Dimensão       (atual: {dim})")
        print("5 - Processar Arquivo")
        print("6 - Processar Pasta")
        print("0 - Sair")
        choice = input("> ").strip()

        if choice == '0':
            break
        elif choice == '1':
            schema = select_schema()
        elif choice == '2':
            strategy = select_strategy()
        elif choice == '3':
            model = select_embedding()
        elif choice == '4':
            dim = select_dimension()
        elif choice == '5':
            path = input("Caminho do arquivo: ").strip()
            start = time.perf_counter()
            process_file(path, strategy, schema, model, dim, results, args.verbose)
            elapsed = time.perf_counter() - start
            print(f"-> Processado em {elapsed:.2f}s (sucesso: {results['processed']}, erros: {results['errors']})")
            input("\nENTER para voltar…")
        elif choice == '6':
            folder = input("Pasta: ").strip()
            all_files = [os.path.join(dp, fn) for dp, _, fns in os.walk(folder)
                         for fn in fns if fn.lower().endswith(('.pdf', '.docx'))]
            total = len(all_files)
            print(f"Total de arquivos a processar: {total}")
            start = time.perf_counter()
            pbar = tqdm(all_files, desc="Processando", unit="arquivo")
            for path in pbar:
                process_file(path, strategy, schema, model, dim, results, args.verbose)
                pbar.set_postfix(processados=results['processed'], erros=results['errors'])
            pbar.close()
            elapsed = time.perf_counter() - start
            print(f"\n=== Resumo ===")
            print(f"Total processados: {results['processed']}")
            print(f"Total de erros: {results['errors']}")
            print(f"Tempo total: {elapsed:.2f}s")
            input("\nENTER para voltar…")
        else:
            input("Opção inválida. ENTER…")

    clear_screen()
    print("\n=== Encerrado ===")
    print(f"Processados: {results['processed']}")
    print(f"Erros: {results['errors']}")

if __name__ == '__main__':
    main()