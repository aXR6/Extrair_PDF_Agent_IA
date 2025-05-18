import os
import logging
import json
import psycopg2
import torch

from adaptive_chunker import hierarchical_chunk, get_sbert_model
from config import PG_HOST, PG_PORT, PG_USER, PG_PASSWORD

# Ajuste para evitar fragmentação de GPU
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def generate_embedding(
    text: str,
    model_name: str,
    dim: int
) -> list[float]:
    """
    Gera embedding usando SBERT local (modelos Hugging Face).
    Tenta GPU e, em caso de CUDA OOM, faz fallback para CPU.
    """
    emb = []
    try:
        sb_model = get_sbert_model(model_name)
        emb_array = sb_model.encode(text, convert_to_numpy=True)
        emb = emb_array.tolist() if hasattr(emb_array, "tolist") else list(emb_array)
    except RuntimeError as e:
        msg = str(e).lower()
        if "cuda out of memory" in msg or "out of memory" in msg:
            logging.warning("CUDA OOM detected. Falling back to CPU.")
            try:
                torch.cuda.empty_cache()
                sb_model = get_sbert_model(model_name)
                emb_array = sb_model.encode(text, convert_to_numpy=True)
                emb = emb_array.tolist() if hasattr(emb_array, "tolist") else list(emb_array)
            except Exception as cpu_e:
                logging.error(f"CPU fallback também falhou: {cpu_e}")
                emb = []
        else:
            logging.error(f"Erro genérico ao gerar embedding: {e}")
            emb = []
    except Exception as e:
        logging.error(f"Erro ao gerar embedding: {e}")
        emb = []

    # Ajusta dimensão: pad/truncate
    if emb and hasattr(emb, '__len__'):
        if len(emb) != dim:
            if len(emb) > dim:
                emb = emb[:dim]
            else:
                emb = emb + [0.0] * (dim - len(emb))
    else:
        emb = [0.0] * dim

    return emb

def save_to_postgres(
    filename: str,
    text: str,
    metadata: dict,
    embedding_model: str,
    embedding_dim: int,
    db_name: str
):
    """
    Conecta ao database 'db_name' dinamicamente e insere cada chunk
    em public.documents (content, metadata JSONB, embedding vector).
    """
    conn = None
    try:
        conn = psycopg2.connect(
            host=PG_HOST,
            port=PG_PORT,
            dbname=db_name,
            user=PG_USER,
            password=PG_PASSWORD
        )
        cur = conn.cursor()
        chunks = hierarchical_chunk(text, metadata)
        logging.info(f"'{filename}' → {len(chunks)} chunks para salvar")

        for idx, chunk in enumerate(chunks):
            clean_chunk = chunk.replace("\x00", "")
            emb = generate_embedding(clean_chunk, embedding_model, embedding_dim)
            record = {
                **metadata,
                "__parent": filename,
                "__chunk_index": idx
            }
            cur.execute(
                "INSERT INTO public.documents (content, metadata, embedding) "
                "VALUES (%s, %s::jsonb, %s)",
                (clean_chunk, json.dumps(record, ensure_ascii=False), emb)
            )

        conn.commit()
        logging.info(f"Dados inseridos em '{db_name}'.")
    except Exception as e:
        logging.error(f"Erro ao salvar em {db_name}: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()