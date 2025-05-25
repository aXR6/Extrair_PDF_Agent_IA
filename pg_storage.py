# pg_storage.py
import os
import logging
import json
import psycopg2
import torch

from adaptive_chunker import hierarchical_chunk, get_cross_encoder, get_sbert_model
from config import PG_HOST, PG_PORT, PG_USER, PG_PASSWORD
from metrics import record_metrics  # decorator

# allow GPU fragmentation mitigation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def generate_embedding(text: str, model_name: str, dim: int) -> list[float]:
    """
    As before—no changes.
    """
    emb = []
    try:
        sb_model = get_sbert_model(model_name)
        arr = sb_model.encode(text, convert_to_numpy=True)
        emb = arr.tolist() if hasattr(arr, "tolist") else list(arr)
    except RuntimeError as e:
        msg = str(e).lower()
        if "out of memory" in msg:
            logging.warning("CUDA OOM, falling back to CPU.")
            try:
                torch.cuda.empty_cache()
                sb_model = get_sbert_model(model_name)
                arr = sb_model.encode(text, convert_to_numpy=True)
                emb = arr.tolist() if hasattr(arr, "tolist") else list(arr)
            except Exception:
                emb = []
        else:
            logging.error(f"Embedding error: {e}")
            emb = []
    except Exception as e:
        logging.error(f"Embedding error: {e}")
        emb = []

    # pad/truncate
    if emb and hasattr(emb, "__len__"):
        if len(emb) != dim:
            if len(emb) > dim:
                emb = emb[:dim]
            else:
                emb += [0.0] * (dim - len(emb))
    else:
        emb = [0.0] * dim

    return emb

def rerank_with_cross_encoder(results: list, query: str, top_k: int = None) -> list:
    """As before."""
    ce = get_cross_encoder()
    pairs = [(query, r['content']) for r in results]
    scores = ce.predict(pairs)
    for r, s in zip(results, scores):
        r['rerank_score'] = float(s)
    sorted_ = sorted(results, key=lambda x: x['rerank_score'], reverse=True)
    return sorted_[:top_k] if top_k else sorted_

@record_metrics
def save_to_postgres(
    filename: str,
    text: str,
    metadata: dict,
    embedding_model: str,
    embedding_dim: int,
    db_name: str
):
    """
    Now: choose chunking model to match embedding_model if it's one of
    our SBERT-based options, else fallback to default SBERT_MODEL_NAME.
    """
    # determine chunking model
    from config import (
        SERAFIM_EMBEDDING_MODEL,
        MPNET_EMBEDDING_MODEL
    )
    if embedding_model in (SERAFIM_EMBEDDING_MODEL, MPNET_EMBEDDING_MODEL):
        chunk_model = embedding_model
    else:
        from config import SBERT_MODEL_NAME
        chunk_model = SBERT_MODEL_NAME

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

        chunks = hierarchical_chunk(text, metadata, chunk_model)
        inserted = []
        logging.info(f"'{filename}' → {len(chunks)} chunks to insert")

        for idx, chunk in enumerate(chunks):
            clean = chunk.replace("\x00", "")
            emb = generate_embedding(clean, embedding_model, embedding_dim)
            rec = {**metadata, "__parent": filename, "__chunk_index": idx}
            cur.execute(
                "INSERT INTO public.documents (content, metadata, embedding) VALUES (%s, %s::jsonb, %s) RETURNING id",
                (clean, json.dumps(rec, ensure_ascii=False), emb)
            )
            did = cur.fetchone()[0]
            inserted.append({'id': did, 'content': clean, 'metadata': rec})

        conn.commit()
        logging.info(f"Inserted into '{db_name}'.")

        # rerank
        reranked = rerank_with_cross_encoder(inserted, metadata.get('__query', ''))
        return reranked

    except Exception as e:
        logging.error(f"Save to Postgres error: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()
