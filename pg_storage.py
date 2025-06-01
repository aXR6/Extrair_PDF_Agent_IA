import os
import logging
import json
import psycopg2
import torch
from adaptive_chunker import hierarchical_chunk, get_sbert_model
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
                     embedding_dim: int):
    conn = None
    try:
        conn = psycopg2.connect(
            host=PG_HOST,
            port=PG_PORT,
            dbname=PG_DATABASE,
            user=PG_USER,
            password=PG_PASSWORD
        )
        cur = conn.cursor()

        chunks = hierarchical_chunk(text, metadata, embedding_model)
        inserted = []
        logging.info(f"'{filename}': {len(chunks)} chunks")

        table = f"public.documents_{embedding_dim}"

        for idx, chunk in enumerate(chunks):
            clean = chunk.replace("\x00", "")
            emb = generate_embedding(clean, embedding_model, embedding_dim)
            rec = {**metadata, "__parent": filename, "__chunk_index": idx}
            cur.execute(
                f"INSERT INTO {table} (content, metadata, embedding) "
                f"VALUES (%s, %s::jsonb, %s) RETURNING id",
                (clean, json.dumps(rec, ensure_ascii=False), emb)
            )
            doc_id = cur.fetchone()[0]
            inserted.append({'id': doc_id, 'content': clean, 'metadata': rec})

        conn.commit()

        # — re-ranking se houver __query
        query = metadata.get('__query', '')
        if query:
            from sentence_transformers import CrossEncoder
            ce = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            pairs = [(query, r['content']) for r in inserted]
            scores = ce.predict(pairs)
            for r, s in zip(inserted, scores):
                r['rerank_score'] = float(s)
            inserted.sort(key=lambda x: x['rerank_score'], reverse=True)

        return inserted

    except Exception as e:
        logging.error(f"Erro saving to Postgres: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()