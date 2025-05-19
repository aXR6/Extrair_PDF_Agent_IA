-- 1. Habilita extensão pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Cria tabela de documentos com tsv_full
CREATE TABLE IF NOT EXISTS public.documents (
  id BIGSERIAL PRIMARY KEY,
  content TEXT NOT NULL,
  metadata JSONB NOT NULL,
  embedding VECTOR(384) NOT NULL,
  tsv_full TSVECTOR
);

-- 3. Função que atualiza tsv_full combinando conteúdo e metadados
CREATE OR REPLACE FUNCTION public.update_tsv_full() RETURNS trigger AS $$
BEGIN
  NEW.tsv_full :=
    setweight(to_tsvector('simple', coalesce(NEW.content, '')), 'A') ||
    setweight(to_tsvector('simple', coalesce(NEW.metadata::text, '')), 'B');
  RETURN NEW;
END
$$ LANGUAGE plpgsql;

-- 4. Trigger para manter tsv_full atualizado
DROP TRIGGER IF EXISTS tsv_full_trigger ON public.documents;
CREATE TRIGGER tsv_full_trigger
  BEFORE INSERT OR UPDATE ON public.documents
  FOR EACH ROW EXECUTE FUNCTION public.update_tsv_full();

-- 5. Índices para performance
-- Vetor: IVFFlat
CREATE INDEX IF NOT EXISTS idx_documents_embedding_ivfflat
  ON public.documents USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 500);

-- Ajusta número de sondagens na sessão de consulta:
-- SET ivfflat.probes = 50;

-- Vetor alternativo: HNSW
CREATE INDEX IF NOT EXISTS idx_documents_embedding_hnsw
  ON public.documents USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 200);

-- Ajusta parâmetro de busca HNSW na sessão:
-- SET hnsw.ef_search = 100;

-- Índice GIN para full-text
CREATE INDEX IF NOT EXISTS idx_documents_tsv_full
  ON public.documents USING gin(tsv_full);

-- 6. Função híbrida avançada de busca por RAG
CREATE OR REPLACE FUNCTION public.match_documents_hybrid(
  query_embedding VECTOR(384),
  query_text      TEXT DEFAULT NULL,
  match_count     INT DEFAULT 5,
  filter          JSONB DEFAULT '{}',
  weight_vec      FLOAT DEFAULT 0.6,
  weight_lex      FLOAT DEFAULT 0.4,
  min_score       FLOAT DEFAULT 0.1,
  pool_multiplier INT DEFAULT 10
) RETURNS TABLE (
  id BIGINT,
  content TEXT,
  metadata JSONB,
  score FLOAT
) LANGUAGE plpgsql AS $$
BEGIN
  RETURN QUERY
  WITH knn_pool AS (
    SELECT
      documents.metadata ->> '__parent' AS parent,
      documents.id, 
      documents.content, 
      documents.metadata, 
      documents.embedding <=> query_embedding AS dist,
      documents.tsv_full
    FROM public.documents
    WHERE documents.metadata @> filter
    ORDER BY documents.embedding <=> query_embedding
    LIMIT match_count * pool_multiplier
  ),
  knn_ts AS (
    SELECT *,
      CASE
        WHEN query_text IS NULL THEN NULL
        WHEN query_text ~ '^".*"$' THEN phraseto_tsquery('simple', trim(both '"' from query_text))
        ELSE websearch_to_tsquery('simple', query_text)
      END AS query_tsq
    FROM knn_pool
  ),
  scored AS (
    SELECT
      knn_ts.parent, knn_ts.id, knn_ts.content, knn_ts.metadata,
      1 - knn_ts.dist AS sim,
      COALESCE(
        CASE 
          WHEN query_tsq IS NOT NULL AND knn_ts.tsv_full @@ query_tsq THEN ts_rank(knn_ts.tsv_full, query_tsq)
          WHEN query_text IS NOT NULL AND knn_ts.content COLLATE "C" LIKE query_text THEN 1
          ELSE 0
        END, 
        0
      ) AS lex_rank
    FROM knn_ts
  ),
  combined AS (
    SELECT
      scored.parent, 
      scored.id, 
      scored.content, 
      scored.metadata,
      CASE
        WHEN scored.lex_rank > 0 THEN weight_vec * scored.sim + weight_lex * scored.lex_rank
        ELSE scored.sim
      END AS score
    FROM scored
  ),
  best_parent AS (
    SELECT combined.parent
    FROM combined
    ORDER BY combined.score DESC
    LIMIT 1
  )
  SELECT combined.id, combined.content, combined.metadata, combined.score
  FROM combined
  WHERE combined.parent = (SELECT best_parent.parent FROM best_parent)
    AND combined.score >= min_score
  ORDER BY combined.score DESC
  LIMIT match_count;
END;
$$;

-- 1) Instale as extensões corretas
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- 2) Índices recomendados (com nomes explícitos)
CREATE INDEX IF NOT EXISTS idx_documents_embedding_ivfflat
  ON public.documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_documents_title_trgm
  ON public.documents USING gist ((metadata->>'title') gist_trgm_ops);

CREATE INDEX IF NOT EXISTS idx_documents_author_trgm
  ON public.documents USING gist ((metadata->>'author') gist_trgm_ops);

CREATE INDEX IF NOT EXISTS idx_documents_subject_trgm
  ON public.documents USING gist ((metadata->>'subject') gist_trgm_ops);

CREATE INDEX IF NOT EXISTS idx_documents_parent_trgm
  ON public.documents USING gist ((metadata->>'__parent') gist_trgm_ops);


-- abaixa o mínimo de similaridade para 0.1 (padrão é 0.3)
SET pg_trgm.similarity_threshold = 0.1;



-- Função híbrida com filtro de similaridade mínima
CREATE OR REPLACE FUNCTION public.match_documents_precise(
  query_embedding VECTOR(384),
  query_text      TEXT     DEFAULT NULL,
  match_count     INT      DEFAULT 5,
  filter          JSONB    DEFAULT '{}',
  weight_vec      FLOAT    DEFAULT 0.6,
  weight_lex      FLOAT    DEFAULT 0.4,
  min_cos_sim     FLOAT    DEFAULT 0.8,  -- similaridade de cosseno mínima
  min_score       FLOAT    DEFAULT 0.5,  -- score combinado mínimo
  pool_multiplier INT      DEFAULT 10
)
RETURNS TABLE (
  id        BIGINT,
  content   TEXT,
  metadata  JSONB,
  score     FLOAT
)
LANGUAGE plpgsql AS $$
DECLARE
  max_cos_dist FLOAT := 1.0 - min_cos_sim;
BEGIN
  RETURN QUERY
  WITH knn_pool AS (
    SELECT
      d.id,
      d.content,
      d.metadata,
      d.embedding <#> query_embedding AS cos_dist,
      d.tsv_full
    FROM public.documents d
    WHERE
      d.metadata @> filter
      AND (d.embedding <#> query_embedding) <= max_cos_dist
    ORDER BY d.embedding <#> query_embedding
    LIMIT match_count * pool_multiplier
  ),
  knn_ts AS (
    SELECT *,
      CASE
        WHEN query_text IS NULL THEN NULL
        WHEN query_text ~ '^".*"$'
          THEN phraseto_tsquery('simple', trim(both '"' from query_text))
        ELSE websearch_to_tsquery('simple', query_text)
      END AS query_tsq
    FROM knn_pool
  ),
  scored AS (
    SELECT
      kt.id,
      kt.content,
      kt.metadata,
      (1 - kt.cos_dist) AS sim,
      COALESCE(
        CASE
          WHEN kt.query_tsq IS NOT NULL AND kt.tsv_full @@ kt.query_tsq
            THEN ts_rank(kt.tsv_full, kt.query_tsq)
          WHEN kt.query_text IS NOT NULL
            AND kt.content ILIKE '%' || kt.query_text || '%'
            THEN 1
          ELSE 0
        END,
        0
      ) AS lex_rank
    FROM knn_ts kt
  ),
  combined AS (
    SELECT
      s.id,
      s.content,
      s.metadata,
      CASE
        WHEN s.lex_rank > 0
          THEN weight_vec * s.sim + weight_lex * s.lex_rank
        ELSE s.sim
      END AS score
    FROM scored s
  )
  SELECT
    c.id,
    c.content,
    c.metadata,
    c.score
  FROM combined c
  WHERE c.score >= min_score
  ORDER BY c.score DESC
  LIMIT match_count;
END;
$$;

-- Uso (p.ex. no n8n):
-- Query Parameters:
-- [ embedding_array, chatInput, 5, {}, 0.6, 0.4, 0.85, 0.7, 10 ]
--
-- Aqui pedimos:
-- • top 5 itens (match_count)
-- • vetor+texto híbrido (pesos 0.6/0.4)
-- • somente cosseno ≥ 0.85 (min_cos_sim)
-- • somente score ≥ 0.7 (min_score)
-- • pool de 50 candidatos (5×10)

-- Não esqueça de ajustar também, em sessão, os parâmetros do índice:
-- SET hnsw.ef_search = 200;            -- para HNSW, mais recall
-- SET ivfflat.probes   = 50;            -- para IVFFlat, mais coesão