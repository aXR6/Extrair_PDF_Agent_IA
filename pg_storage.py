import os
import logging
import json
import psycopg2
import torch

from adaptive_chunker import hierarchical_chunk, get_cross_encoder, get_sbert_model
from config import PG_HOST, PG_PORT, PG_USER, PG_PASSWORD
from metrics import record_metrics  # Decorator de métricas

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

def rerank_with_cross_encoder(results: list, query: str, top_k: int = None) -> list:
    """
    Re-rank os documentos usando um modelo cross-encoder para maior precisão.
    """
    ce = get_cross_encoder()
    pairs = [(query, r['content']) for r in results]
    scores = ce.predict(pairs)
    for r, s in zip(results, scores):
        r['rerank_score'] = float(s)
    ranked = sorted(results, key=lambda x: x['rerank_score'], reverse=True)
    return ranked[:top_k] if top_k else ranked

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
    Conecta ao PostgreSQL e insere cada chunk em public.documents,
    depois reordena com cross-encoder e coleta métricas.
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

        # Chunking semântico (usa hierarchical_chunk, que engloba sections e sliding window)
        chunks = hierarchical_chunk(text, metadata)
        inserted = []
        logging.info(f"'{filename}' → {len(chunks)} chunks para salvar")

        # Inserção de cada chunk
        for idx, chunk in enumerate(chunks):
            clean_chunk = chunk.replace("\x00", "")
            emb = generate_embedding(clean_chunk, embedding_model, embedding_dim)
            rec = {**metadata, "__parent": filename, "__chunk_index": idx}
            cur.execute(
                "INSERT INTO public.documents (content, metadata, embedding) VALUES (%s, %s::jsonb, %s) RETURNING id",
                (clean_chunk, json.dumps(rec, ensure_ascii=False), emb)
            )
            doc_id = cur.fetchone()[0]
            inserted.append({'id': doc_id, 'content': clean_chunk, 'metadata': rec})

        conn.commit()
        logging.info(f"Dados inseridos em '{db_name}'.")

        # Re-ranking usando cross-encoder
        query_text = metadata.get('__query', '')
        reranked = rerank_with_cross_encoder(inserted, query_text)
        logging.info(f"Inseridos {len(inserted)} chunks; retornando {len(reranked)} após re-ranking.")

        return reranked

    except Exception as e:
        logging.error(f"Erro ao salvar em {db_name}: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()
