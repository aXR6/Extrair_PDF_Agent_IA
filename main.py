#!/usr/bin/env python3
import os
import sys
import argparse
import time
import logging
from tqdm import tqdm

# Garante imports locais
sys.path.insert(0, os.path.dirname(__file__))

from config import (
    OLLAMA_EMBEDDING_MODEL, SERAFIM_EMBEDDING_MODEL,
    MINILM_L6_V2, MINILM_L12_V2, MPNET_EMBEDDING_MODEL,
    DIM_MXBAI, DIM_SERAFIM, DIM_MINILM_L6, DIM_MINIL12, DIM_MPNET,
    OCR_THRESHOLD, validate_config
)
from extractors import extract_text
from utils import setup_logging, is_valid_file, build_record
from pg_storage import save_to_postgres
from adaptive_chunker import get_sbert_model

# Valida configuração e inicializa SBERT + logs
validate_config()
setup_logging()
get_sbert_model()  # Carrega SBERT em CPU logo no início

# Opções de menu
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


def select_strategy(current: str) -> str:
    print("\n*** Selecione Estratégia ***")
    for i, k in enumerate(STRATEGY_OPTIONS, 1):
        print(f"{i} - {k}")
    c = input(f"Escolha [{current}]: ").strip()
    if c.isdigit() and 1 <= int(c) <= len(STRATEGY_OPTIONS):
        return STRATEGY_OPTIONS[int(c)-1]
    return current


def select_embedding(current: str) -> str:
    print("\n*** Selecione Embedding ***")
    for k, n in EMBED_MODELS.items():
        print(f"{k} - {n}")
    c = input(f"Escolha [{current}]: ").strip()
    return EMBED_MODELS.get(c, current)


def select_dimension(current: int) -> int:
    print("\n*** Selecione Dimensão ***")
    for k, d in DIMENSIONS.items():
        print(f"{k} - {d}")
    c = input(f"Escolha [{current}]: ").strip()
    return DIMENSIONS.get(c, current)


def process_file(path: str, strat: str, model: str, dim: int, stats: dict):
    """
    Processa um único arquivo: extrai texto, gera embeddings e salva no PostgreSQL.
    """
    filename = os.path.basename(path)
    logging.info(f"→ Processando arquivo: {filename}  |  Estratégia: {strat}  |  Embedding: {model}  |  Dimensão: {dim}")

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
        logging.error(f"Arquivo inválido: {filename}")
        return

    text = extract_text(p2, strat)
    if not text or len(text.strip()) < OCR_THRESHOLD:
        logging.error(f"Não foi possível extrair texto: {filename}")
        stats['errors'] += 1
        return

    rec = build_record(p2, text)
    try:
        inserted_count = save_to_postgres(
            filename, rec['text'], rec['info'],
            model, dim
        )
        stats['processed'] += 1
        logging.info(f"→ '{filename}' inseriu {inserted_count} chunks no banco.")
    except Exception as e:
        logging.error(f"Erro salvando '{filename}': {e}")
        stats['errors'] += 1
    finally:
        # Forçar remoção de textos e metadados grandes e coletar lixo
        try:
            del text
            del rec
            import gc; gc.collect()
        except Exception:
            pass


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
            strat = select_strategy(strat)

        elif c == "2":
            model = select_embedding(model)

        elif c == "3":
            dim = select_dimension(dim)

        elif c == "4":
            # Modo “Arquivo”: processa apenas um PDF
            f = input("Arquivo: ").strip()
            if not f:
                print("Nenhum arquivo informado.")
                time.sleep(1)
                continue

            start = time.perf_counter()
            process_file(f, strat, model, dim, stats)
            dt = time.perf_counter() - start

            print(f"\n→ Tempo gasto: {dt:.2f}s  •  Processados: {stats['processed']}  •  Erros: {stats['errors']}")
            input("ENTER para continuar…")

        elif c == "5":
            # Modo “Pasta”: varre todos os arquivos de dentro de um diretório
            d = input("Pasta: ").strip()
            if not d or not os.path.isdir(d):
                print("Pasta inválida ou não existe.")
                time.sleep(1)
                continue

            # Coleta de todos os PDFs, DOCX e Imagens na pasta recursivamente
            files = [
                os.path.join(root, fname)
                for root, _, files_ in os.walk(d)
                for fname in files_
                if fname.lower().endswith((".pdf", ".docx", ".png", ".jpg", ".jpeg", ".tiff"))
            ]

            total_files = len(files)
            print(f"Total de arquivos encontrados: {total_files}")
            if total_files == 0:
                input("ENTER para continuar…")
                continue

            start = time.perf_counter()

            # tqdm com descrição dinâmica do arquivo atual
            pbar = tqdm(files, unit="arquivo")
            for path in pbar:
                basename = os.path.basename(path)
                # Altera a descrição para mostrar exatamente qual arquivo está sendo processado
                pbar.set_description(f"Processando → {basename} | Strat: {strat} | Emb: {model} | Dim: {dim}")
                process_file(path, strat, model, dim, stats)
                pbar.set_postfix({"P": stats['processed'], "E": stats['errors']})
                # Coleta lixo após cada arquivo
                try:
                    import gc; gc.collect()
                except Exception:
                    pass

            pbar.close()
            dt = time.perf_counter() - start

            print(f"\n=== Resumo final ===")
            print(f"  Processados: {stats['processed']}  •  Erros: {stats['errors']}  •  Tempo total: {dt:.2f}s")
            input("ENTER para continuar…")

        else:
            print("Opção inválida.")
            time.sleep(1)

    clear_screen()
    print(f"Processados: {stats['processed']}  •  Erros: {stats['errors']}")


if __name__ == "__main__":
    main()