# pg_storage.py
import os
import logging
import json
import psycopg2
import torch
from adaptive_chunker import hierarchical_chunk_generator, get_sbert_model
from sentence_transformers import CrossEncoder
from config import PG_HOST, PG_PORT, PG_USER, PG_PASSWORD, PG_DATABASE
from metrics import record_metrics

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

_CE_CACHE: dict = {}

def get_cross_encoder(model_name: str, device: str) -> CrossEncoder:
    """Retorna CrossEncoder em cache para o dispositivo escolhido."""
    key = (model_name, device)
    if key not in _CE_CACHE:
        try:
            logging.info(f"Carregando CrossEncoder '{model_name}' em {device}…")
            _CE_CACHE[key] = CrossEncoder(model_name, device=device)
            logging.info(f"CrossEncoder '{model_name}' carregado com sucesso em {device}.")
        except Exception as e:
            logging.error(f"Falha ao carregar CrossEncoder '{model_name}' em {device}: {e}")
            raise
    return _CE_CACHE[key]

def generate_embedding(text: str, model_name: str, dim: int, device: str) -> list[float]:
    """Gera embedding no dispositivo escolhido com fallback para CPU."""
    try:
        model = get_sbert_model(model_name, device=device)
        # Garante modo inference (sem gradiente)
        with torch.no_grad():
            emb = model.encode(text, convert_to_numpy=True)
    except RuntimeError as e:
        msg = str(e).lower()
        if "out of memory" in msg:
            logging.warning("CUDA OOM – tentando em CPU")
            torch.cuda.empty_cache()
            model = get_sbert_model(model_name, device="cpu")
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
                     embedding_dim: int,
                     device: str) -> list[dict]:
    """
    Insere no PostgreSQL cada chunk gerado em streaming pelo hierarchical_chunk_generator.
    Retorna uma lista de dicionários contendo:
      - 'id': id gerado pelo banco para cada chunk,
      - 'content': texto do chunk,
      - 'metadata': metadados JSONB originais + __parent e __chunk_index.
    Após inserir todos os chunks, se houver chave '__query' em metadata, executa re-ranking
    com CrossEncoder e adiciona o campo 'rerank_score' em cada dicionário antes de ordenar.
    """
    conn = None
    inserted = []

    if device == "auto":
        device_use = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "gpu":
        if torch.cuda.is_available():
            device_use = "cuda"
        else:
            logging.warning("GPU selecionada, mas não disponível. Usando CPU.")
            device_use = "cpu"
    else:
        device_use = "cpu"

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

        # Inserção em streaming: consome o gerador de chunks
        for idx, chunk in enumerate(hierarchical_chunk_generator(text, metadata, embedding_model, device_use)):
            clean = chunk.replace("\x00", "")
            emb = generate_embedding(clean, embedding_model, embedding_dim, device_use)

            # Metadata mantém todas as chaves originais + __parent e __chunk_index
            rec = {**metadata, "__parent": filename, "__chunk_index": idx}

            cur.execute(
                f"INSERT INTO {table} (content, metadata, embedding) "
                f"VALUES (%s, %s::jsonb, %s) RETURNING id",
                (clean, json.dumps(rec, ensure_ascii=False), emb)
            )
            doc_id = cur.fetchone()[0]
            inserted.append({'id': doc_id, 'content': clean, 'metadata': rec})

            # Limpeza imediata de objetos pesados
            del clean
            del emb
            del rec
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        conn.commit()

        # — Re‐ranking com CrossEncoder se existir __query —
        query = metadata.get('__query', '')
        if query:
            ce = get_cross_encoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device_use)
            pairs = [(query, r['content']) for r in inserted]
            scores = ce.predict(pairs)
            for r, s in zip(inserted, scores):
                r['rerank_score'] = float(s)
            # Ordena pela pontuação de re-ranking (maior para menor)
            inserted.sort(key=lambda x: x.get('rerank_score', 0.0), reverse=True)

            # Limpa também cache do CrossEncoder
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        return inserted

    except Exception as e:
        logging.error(f"Erro saving to Postgres: {e}")
        if conn:
            conn.rollback()
        raise

    finally:
        if conn:
            conn.close()
