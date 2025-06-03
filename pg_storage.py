# pg_storage.py

import os
import logging
import json
import psycopg2
import torch
from adaptive_chunker import hierarchical_chunk_generator, get_sbert_model
from config import PG_HOST, PG_PORT, PG_USER, PG_PASSWORD, PG_DATABASE
from metrics import record_metrics

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

def generate_embedding(text: str, model_name: str, dim: int) -> list[float]:
    """Gera embedding com fallback CPU e liberação de memória."""
    try:
        model = get_sbert_model(model_name)
        # Garante modo inference (sem gradiente)
        with torch.no_grad():
            emb = model.encode(text, convert_to_numpy=True)
    except RuntimeError as e:
        msg = str(e).lower()
        if "out of memory" in msg:
            logging.warning("CUDA OOM – tentando em CPU")
            torch.cuda.empty_cache()
            model = get_sbert_model(model_name)
            with torch.no_grad():
                emb = model.encode(text, convert_to_numpy=True)
        else:
            logging.error(f"Erro embed genérico: {e}")
            return [0.0] * dim
    except Exception as e:
        logging.error(f"Erro ao gerar embedding: {e}")
        return [0.0] * dim

    vec = emb.tolist() if hasattr(emb, "tolist") else list(emb)
    # Ajusta comprimento para a dimensão correta
    if len(vec) < dim:
        vec += [0.0] * (dim - len(vec))
    elif len(vec) > dim:
        vec = vec[:dim]

    # Limpa cache da GPU (precaução)
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    return vec

@record_metrics
def save_to_postgres(filename: str,
                     text: str,
                     metadata: dict,
                     embedding_model: str,
                     embedding_dim: int) -> int:
    """
    Insere no PostgreSQL cada chunk gerado em streaming pelo hierarchical_chunk_generator.
    Retorna o número de chunks inseridos (ou raise em caso de erro).
    """
    conn = None
    total_inserted = 0

    try:
        conn = psycopg2.connect(
            host=PG_HOST,
            port=PG_PORT,
            dbname=PG_DATABASE,
            user=PG_USER,
            password=PG_PASSWORD
        )
        cur = conn.cursor()

        table = f"public.documents_{embedding_dim}"

        # Consumir o generator em streaming: não acumulamos lista de chunks
        for idx, chunk in enumerate(hierarchical_chunk_generator(text, metadata, embedding_model)):
            clean = chunk.replace("\x00", "")
            emb = generate_embedding(clean, embedding_model, embedding_dim)

            # Metadata mantém todas as chaves originais + __parent e __chunk_index
            rec = {**metadata, "__parent": filename, "__chunk_index": idx}

            cur.execute(
                f"INSERT INTO {table} (content, metadata, embedding) "
                f"VALUES (%s, %s::jsonb, %s) RETURNING id",
                (clean, json.dumps(rec, ensure_ascii=False), emb)
            )
            _ = cur.fetchone()[0]
            total_inserted += 1

            # Liberar imediatamente variáveis de uso pesado
            del clean
            del emb
            del rec
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        conn.commit()
        return total_inserted

    except Exception as e:
        logging.error(f"Erro saving to Postgres: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()