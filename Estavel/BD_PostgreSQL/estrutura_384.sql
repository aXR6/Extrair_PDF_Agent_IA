-- =============================================================================
-- Estrutura Completa de Banco para Busca Semântica Híbrida (RAG)
-- Inclui instruções originais e melhorias recomendadas
-- =============================================================================

-- 1) Extensões Necessárias
CREATE EXTENSION IF NOT EXISTS vector;      -- pgvector para embeddings
CREATE EXTENSION IF NOT EXISTS pg_trgm;      -- trigramas para fuzzy text filtering

-- 2) Tabela de Documentos
CREATE TABLE IF NOT EXISTS public.documents (
  id        BIGSERIAL PRIMARY KEY,
  content   TEXT         NOT NULL,
  metadata  JSONB        NOT NULL,
  embedding VECTOR(384)  NOT NULL,
  tsv_full  TSVECTOR
);

-- (Opcional) Exemplo de particionamento mensal para alta escala
-- CREATE TABLE public.documents_2025_05 (
--   LIKE public.documents INCLUDING ALL
-- ) PARTITION OF public.documents
--   FOR VALUES FROM ('2025-05-01') TO ('2025-06-01');

-- 3) Função e Trigger para manter tsv_full atualizado
CREATE OR REPLACE FUNCTION public.update_tsv_full() RETURNS trigger AS $$
BEGIN
  NEW.tsv_full :=
    -- Full-text híbrido: peso A para content, peso B para metadata
    setweight(to_tsvector('simple', coalesce(NEW.content, '')), 'A') ||
    setweight(to_tsvector('simple', coalesce(NEW.metadata::text, '')), 'B');
  RETURN NEW;
END
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS tsv_full_trigger ON public.documents;
CREATE TRIGGER tsv_full_trigger
  BEFORE INSERT OR UPDATE ON public.documents
  FOR EACH ROW EXECUTE FUNCTION public.update_tsv_full();

-- 4) Índices Vetoriais
-- 4.1) HNSW (padrão primário para RAG)
CREATE INDEX IF NOT EXISTS idx_documents_embedding_hnsw
  ON public.documents USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 200);

-- 4.2) IVFFlat (fallback opcional)
CREATE INDEX IF NOT EXISTS idx_documents_embedding_ivfflat
  ON public.documents USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 500);

-- 5) Índices Full-Text & Trigramas
-- 5.1) GIN para tsv_full
CREATE INDEX IF NOT EXISTS idx_documents_tsv_full
  ON public.documents USING gin(tsv_full);

-- 5.2) GIN global para JSONB metadata
CREATE INDEX IF NOT EXISTS idx_documents_metadata_gin
  ON public.documents USING gin(metadata);

-- 5.3) GiST + pg_trgm em campos críticos de metadata
CREATE INDEX IF NOT EXISTS idx_documents_title_trgm
  ON public.documents USING gist ((metadata->>'title') gist_trgm_ops);

CREATE INDEX IF NOT EXISTS idx_documents_author_trgm
  ON public.documents USING gist ((metadata->>'author') gist_trgm_ops);

-- Exemplo de índice parcial para um campo específico
CREATE INDEX IF NOT EXISTS idx_documents_type_trgm
  ON public.documents USING gist ((metadata->>'type') gist_trgm_ops)
  WHERE metadata ? 'type';

-- 6) Text Search Dictionary para sinônimos (opcional)
-- 6.1) Defina seu arquivo synonyms.txt e carregue em um dicionário
-- CREATE TEXT SEARCH DICTIONARY syn_dict (
--   TEMPLATE = pg_catalog.synonym,
--   SYNONYMS = english_synonyms
-- );
--
-- 6.2) Configure uma configuração customizada
-- CREATE TEXT SEARCH CONFIGURATION syn_config (COPY = simple);
-- ALTER TEXT SEARCH CONFIGURATION syn_config
--   ALTER MAPPING FOR asciiword WITH syn_dict, simple;
--
-- 6.3) Aplique no trigger se desejar usar sinônimos
-- setweight(to_tsvector('syn_config', NEW.content), 'A') || …

-- 7) Parâmetros de Sessão Recomendados
-- (execute antes de cada consulta RAG)
-- SET hnsw.ef_search = 200;
-- SET ivfflat.probes   = 50;
-- SET pg_trgm.similarity_threshold = 0.1;

-- 8) Função Híbrida Avançada de Busca RAG (match_documents_hybrid)
CREATE OR REPLACE FUNCTION public.match_documents_hybrid(
  query_embedding VECTOR(384),
  query_text      TEXT     DEFAULT NULL,
  match_count     INT      DEFAULT 5,
  filter          JSONB    DEFAULT '{}',
  weight_vec      FLOAT    DEFAULT 0.6,
  weight_lex      FLOAT    DEFAULT 0.4,
  min_score       FLOAT    DEFAULT 0.1,
  pool_multiplier INT      DEFAULT 10
) RETURNS TABLE (
  id        BIGINT,
  content   TEXT,
  metadata  JSONB,
  score     FLOAT
) LANGUAGE plpgsql AS $$
BEGIN
  RETURN QUERY
  WITH knn_pool AS (
    SELECT
      d.metadata ->> '__parent' AS parent,
      d.id, d.content, d.metadata,
      d.embedding <=> query_embedding AS dist,
      d.tsv_full
    FROM public.documents d
    WHERE d.metadata @> filter
    ORDER BY d.embedding <=> query_embedding
    LIMIT match_count * pool_multiplier
  ),
  knn_ts AS (
    SELECT *,
      CASE
        WHEN query_text IS NULL THEN NULL
        WHEN query_text ~ '^\".*\"$' THEN phraseto_tsquery('simple', trim(both '\"' from query_text))
        ELSE websearch_to_tsquery('simple', query_text)
      END AS query_tsq
    FROM knn_pool
  ),
  scored AS (
    SELECT
      kt.parent, kt.id, kt.content, kt.metadata,
      1 - kt.dist AS sim,
      COALESCE(
        CASE
          WHEN kt.query_tsq IS NOT NULL AND kt.tsv_full @@ kt.query_tsq
            THEN ts_rank(kt.tsv_full, kt.query_tsq)
          WHEN query_text IS NOT NULL AND kt.content ILIKE '%' || query_text || '%'
            THEN 1
          ELSE 0
        END, 0
      ) AS lex_rank
    FROM knn_ts kt
  ),
  combined AS (
    SELECT
      s.parent, s.id, s.content, s.metadata,
      CASE
        WHEN s.lex_rank > 0 THEN weight_vec * s.sim + weight_lex * s.lex_rank
        ELSE s.sim
      END AS score
    FROM scored s
  ),
  best_parent AS (
    SELECT parent
    FROM combined
    ORDER BY score DESC
    LIMIT 1
  )
  SELECT c.id, c.content, c.metadata, c.score
  FROM combined c
  WHERE c.parent = (SELECT parent FROM best_parent)
    AND c.score >= min_score
  ORDER BY c.score DESC
  LIMIT match_count;
END;
$$;

-- 9) Função Precise com filtro mínimo de cosseno (match_documents_precise)
CREATE OR REPLACE FUNCTION public.match_documents_precise(
  query_embedding VECTOR(384),
  query_text      TEXT     DEFAULT NULL,
  match_count     INT      DEFAULT 5,
  filter          JSONB    DEFAULT '{}',
  weight_vec      FLOAT    DEFAULT 0.6,
  weight_lex      FLOAT    DEFAULT 0.4,
  min_cos_sim     FLOAT    DEFAULT 0.8,
  min_score       FLOAT    DEFAULT 0.5,
  pool_multiplier INT      DEFAULT 10
) RETURNS TABLE (
  id        BIGINT,
  content   TEXT,
  metadata  JSONB,
  score     FLOAT
) LANGUAGE plpgsql AS $$
DECLARE
  max_cos_dist FLOAT := 1.0 - min_cos_sim;
BEGIN
  RETURN QUERY
  WITH knn_pool AS (
    SELECT
      d.id, d.content, d.metadata,
      d.embedding <#> query_embedding AS cos_dist,
      d.tsv_full
    FROM public.documents d
    WHERE d.metadata @> filter
      AND d.embedding <#> query_embedding <= max_cos_dist
    ORDER BY d.embedding <#> query_embedding
    LIMIT match_count * pool_multiplier
  ),
  knn_ts AS (
    SELECT *,
      CASE
        WHEN query_text IS NULL THEN NULL
        WHEN query_text ~ '^\".*\"$' THEN phraseto_tsquery('simple', trim(both '\"' from query_text))
        ELSE websearch_to_tsquery('simple', query_text)
      END AS query_tsq
    FROM knn_pool
  ),
  scored AS (
    SELECT
      kt.id, kt.content, kt.metadata,
      (1 - kt.cos_dist) AS sim,
      COALESCE(
        CASE
          WHEN kt.query_tsq IS NOT NULL AND kt.tsv_full @@ kt.query_tsq
            THEN ts_rank(kt.tsv_full, kt.query_tsq)
          WHEN query_text IS NOT NULL AND kt.content ILIKE '%' || query_text || '%'
            THEN 1
          ELSE 0
        END, 0
      ) AS lex_rank
    FROM knn_ts kt
  ),
  combined AS (
    SELECT
      s.id, s.content, s.metadata,
      CASE
        WHEN s.lex_rank > 0 THEN weight_vec * s.sim + weight_lex * s.lex_rank
        ELSE s.sim
      END AS score
    FROM scored s
  )
  SELECT c.id, c.content, c.metadata, c.score
  FROM combined c
  WHERE c.score >= min_score
  ORDER BY c.score DESC
  LIMIT match_count;
END;
$$;

-- -----------------------------------------------------------------------------
-- FIM DA ESTRUTURA COMPLETA
-- -----------------------------------------------------------------------------