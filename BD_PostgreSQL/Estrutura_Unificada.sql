CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

DROP TEXT SEARCH CONFIGURATION IF EXISTS public.pt_en;
DROP TEXT SEARCH DICTIONARY IF EXISTS public.pt_stem;
DROP TEXT SEARCH DICTIONARY IF EXISTS public.eng_stem;

CREATE TEXT SEARCH DICTIONARY public.pt_stem (
  TEMPLATE = snowball,
  LANGUAGE = portuguese
);
CREATE TEXT SEARCH DICTIONARY public.eng_stem (
  TEMPLATE = snowball,
  LANGUAGE = english
);

CREATE TEXT SEARCH CONFIGURATION public.pt_en (COPY = pg_catalog.simple);
ALTER TEXT SEARCH CONFIGURATION public.pt_en
  ALTER MAPPING FOR asciiword, asciihword, hword_asciipart
    WITH public.pt_stem, public.eng_stem, simple;

CREATE OR REPLACE FUNCTION public.update_tsv_full() RETURNS trigger AS $$
BEGIN
  NEW.tsv_full :=
    setweight(to_tsvector('public.pt_en', COALESCE(NEW.content, '')), 'A')
    || setweight(to_tsvector('public.pt_en', COALESCE(NEW.metadata::text, '')), 'B');
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TABLE IF NOT EXISTS public.documents_384 (
  id         BIGSERIAL    PRIMARY KEY,
  content    TEXT         NOT NULL,
  metadata   JSONB        NOT NULL,
  embedding  VECTOR(384)  NOT NULL,
  tsv_full   TSVECTOR,
  created_at TIMESTAMPTZ  DEFAULT now()
);
DROP TRIGGER IF EXISTS tsv_full_trigger ON public.documents_384;
CREATE TRIGGER tsv_full_trigger
  BEFORE INSERT OR UPDATE ON public.documents_384
  FOR EACH ROW EXECUTE FUNCTION public.update_tsv_full();
CREATE INDEX IF NOT EXISTS idx_docs384_emb_hnsw ON public.documents_384 USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 200);
CREATE INDEX IF NOT EXISTS idx_docs384_emb_ivf ON public.documents_384 USING ivfflat (embedding vector_cosine_ops) WITH (lists = 400);
CREATE INDEX IF NOT EXISTS idx_docs384_tsv ON public.documents_384 USING gin(tsv_full);
CREATE INDEX IF NOT EXISTS idx_docs384_meta ON public.documents_384 USING gin(metadata);
CREATE INDEX IF NOT EXISTS idx_docs384_title ON public.documents_384 USING gin((metadata->>'title') gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_docs384_auth ON public.documents_384 USING gin((metadata->>'author') gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_docs384_type ON public.documents_384 USING gin((metadata->>'type') gin_trgm_ops) WHERE metadata ? 'type';
CREATE INDEX IF NOT EXISTS idx_docs384_parent ON public.documents_384 USING gin((metadata->>'__parent') gin_trgm_ops);

CREATE TABLE IF NOT EXISTS public.documents_768 (LIKE public.documents_384 INCLUDING ALL);
ALTER TABLE public.documents_768 ALTER COLUMN embedding TYPE VECTOR(768);
DROP TRIGGER IF EXISTS tsv_full_trigger ON public.documents_768;
CREATE TRIGGER tsv_full_trigger
  BEFORE INSERT OR UPDATE ON public.documents_768
  FOR EACH ROW EXECUTE FUNCTION public.update_tsv_full();
CREATE INDEX IF NOT EXISTS idx_docs768_emb_hnsw ON public.documents_768 USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 200);
CREATE INDEX IF NOT EXISTS idx_docs768_emb_ivf ON public.documents_768 USING ivfflat (embedding vector_cosine_ops) WITH (lists = 400);
CREATE INDEX IF NOT EXISTS idx_docs768_tsv ON public.documents_768 USING gin(tsv_full);
CREATE INDEX IF NOT EXISTS idx_docs768_meta ON public.documents_768 USING gin(metadata);
CREATE INDEX IF NOT EXISTS idx_docs768_title ON public.documents_768 USING gin((metadata->>'title') gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_docs768_auth ON public.documents_768 USING gin((metadata->>'author') gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_docs768_type ON public.documents_768 USING gin((metadata->>'type') gin_trgm_ops) WHERE metadata ? 'type';
CREATE INDEX IF NOT EXISTS idx_docs768_parent ON public.documents_768 USING gin((metadata->>'__parent') gin_trgm_ops);

CREATE TABLE IF NOT EXISTS public.documents_1024 (LIKE public.documents_384 INCLUDING ALL);
ALTER TABLE public.documents_1024 ALTER COLUMN embedding TYPE VECTOR(1024);
DROP TRIGGER IF EXISTS tsv_full_trigger ON public.documents_1024;
CREATE TRIGGER tsv_full_trigger
  BEFORE INSERT OR UPDATE ON public.documents_1024
  FOR EACH ROW EXECUTE FUNCTION public.update_tsv_full();
CREATE INDEX IF NOT EXISTS idx_docs1024_emb_hnsw ON public.documents_1024 USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 200);
CREATE INDEX IF NOT EXISTS idx_docs1024_emb_ivf ON public.documents_1024 USING ivfflat (embedding vector_cosine_ops) WITH (lists = 400);
CREATE INDEX IF NOT EXISTS idx_docs1024_tsv ON public.documents_1024 USING gin(tsv_full);
CREATE INDEX IF NOT EXISTS idx_docs1024_meta ON public.documents_1024 USING gin(metadata);
CREATE INDEX IF NOT EXISTS idx_docs1024_title ON public.documents_1024 USING gin((metadata->>'title') gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_docs1024_auth ON public.documents_1024 USING gin((metadata->>'author') gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_docs1024_type ON public.documents_1024 USING gin((metadata->>'type') gin_trgm_ops) WHERE metadata ? 'type';
CREATE INDEX IF NOT EXISTS idx_docs1024_parent ON public.documents_1024 USING gin((metadata->>'__parent') gin_trgm_ops);

CREATE TABLE IF NOT EXISTS public.documents_1536 (LIKE public.documents_384 INCLUDING ALL);
ALTER TABLE public.documents_1536 ALTER COLUMN embedding TYPE VECTOR(1536);
DROP TRIGGER IF EXISTS tsv_full_trigger ON public.documents_1536;
CREATE TRIGGER tsv_full_trigger
  BEFORE INSERT OR UPDATE ON public.documents_1536
  FOR EACH ROW EXECUTE FUNCTION public.update_tsv_full();
CREATE INDEX IF NOT EXISTS idx_docs1536_emb_hnsw ON public.documents_1536 USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 200);
CREATE INDEX IF NOT EXISTS idx_docs1536_emb_ivf ON public.documents_1536 USING ivfflat (embedding vector_cosine_ops) WITH (lists = 400);
CREATE INDEX IF NOT EXISTS idx_docs1536_tsv ON public.documents_1536 USING gin(tsv_full);
CREATE INDEX IF NOT EXISTS idx_docs1536_meta ON public.documents_1536 USING gin(metadata);
CREATE INDEX IF NOT EXISTS idx_docs1536_title ON public.documents_1536 USING gin((metadata->>'title') gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_docs1536_auth ON public.documents_1536 USING gin((metadata->>'author') gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_docs1536_type ON public.documents_1536 USING gin((metadata->>'type') gin_trgm_ops) WHERE metadata ? 'type';
CREATE INDEX IF NOT EXISTS idx_docs1536_parent ON public.documents_1536 USING gin((metadata->>'__parent') gin_trgm_ops);

CREATE OR REPLACE FUNCTION public.match_documents_hybrid(
  query_embedding VECTOR,
  query_text      TEXT      DEFAULT NULL,
  match_count     INT       DEFAULT 5,
  filter          JSONB     DEFAULT '{}'::jsonb,
  weight_vec      FLOAT     DEFAULT 0.6,
  weight_lex      FLOAT     DEFAULT 0.4,
  min_score       FLOAT     DEFAULT 0.1,
  pool_multiplier INT       DEFAULT 10
) RETURNS TABLE (
  id       BIGINT,
  content  TEXT,
  metadata JSONB,
  score    FLOAT
) LANGUAGE plpgsql AS $$
DECLARE
  dim      INT   := vector_dims(query_embedding);
  tbl_full TEXT  := format('%I.%I', 'public', 'documents_' || dim);
  sql      TEXT;
BEGIN
  IF dim NOT IN (384,768,1024,1536) THEN
    RAISE EXCEPTION 'Dimensão não suportada: %', dim;
  END IF;
  sql := format($body$
    WITH knn_pool AS (
      SELECT d.metadata->>'__parent' AS parent, d.id, d.content, d.metadata,
             d.embedding <=> $1 AS dist, d.tsv_full
      FROM %s AS d
      WHERE d.metadata @> $4
      ORDER BY d.embedding <=> $1
      LIMIT $3 * $8
    ),
    knn_ts AS (
      SELECT kp.*, CASE
        WHEN $2 IS NULL THEN NULL
        WHEN $2 ~ '^".*"$' THEN phraseto_tsquery('public.pt_en', trim(both '\"' FROM $2))
        ELSE websearch_to_tsquery('public.pt_en', $2)
      END AS tsq
      FROM knn_pool AS kp
    ),
    scored AS (
      SELECT id, content, metadata, 1 - dist AS sim,
             COALESCE(
               CASE
                 WHEN tsq IS NOT NULL AND tsv_full @@ tsq THEN ts_rank(tsv_full, tsq)
                 WHEN $2 IS NOT NULL AND content ILIKE '%%' || $2 || '%%' THEN 1
                 ELSE 0
               END, 0) AS lex_rank
      FROM knn_ts
    ),
    combined AS (
      SELECT id, content, metadata,
             CASE WHEN lex_rank > 0 THEN $5 * sim + $6 * lex_rank ELSE sim END AS score
      FROM scored
    )
    SELECT id, content, metadata, score
    FROM combined
    WHERE score >= $7
    ORDER BY score DESC
    LIMIT $3;
  $body$, tbl_full);
  RETURN QUERY EXECUTE sql
    USING query_embedding, query_text, match_count, filter,
          weight_vec, weight_lex, min_score, pool_multiplier;
END;
$$;

CREATE OR REPLACE FUNCTION public.match_documents_precise(
  query_embedding VECTOR,
  query_text      TEXT      DEFAULT NULL,
  match_count     INT       DEFAULT 5,
  filter          JSONB     DEFAULT '{}'::jsonb,
  weight_vec      FLOAT     DEFAULT 0.6,
  weight_lex      FLOAT     DEFAULT 0.4,
  min_cos_sim     FLOAT     DEFAULT 0.80,
  min_score       FLOAT     DEFAULT 0.50,
  pool_multiplier INT       DEFAULT 10
) RETURNS TABLE (
  id       BIGINT,
  content  TEXT,
  metadata JSONB,
  score    FLOAT
) LANGUAGE plpgsql AS $$
DECLARE
  dim      INT   := vector_dims(query_embedding);
  tbl_full TEXT  := format('%I.%I', 'public', 'documents_' || dim);
  sql      TEXT;
BEGIN
  IF dim NOT IN (384,768,1024,1536) THEN
    RAISE EXCEPTION 'Dimensão não suportada: %', dim;
  END IF;
  sql := format($body$
    WITH knn_pool AS (
      SELECT d.id, d.content, d.metadata, d.embedding <#> $1 AS cos_dist, d.tsv_full
      FROM %s AS d
      WHERE d.metadata @> $4
        AND d.embedding <#> $1 <= 1.0 - $7
      ORDER BY d.embedding <#> $1
      LIMIT $3 * $9
    ),
    knn_ts AS (
      SELECT kp.*, CASE
        WHEN $2 IS NULL THEN NULL
        WHEN $2 ~ '^".*"$' THEN phraseto_tsquery('public.pt_en', trim(both '\"' FROM $2))
        ELSE websearch_to_tsquery('public.pt_en', $2)
      END AS tsq
      FROM knn_pool AS kp
    ),
    scored AS (
      SELECT id, content, metadata, (1 - cos_dist) AS sim,
             COALESCE(
               CASE
                 WHEN tsq IS NOT NULL AND tsv_full @@ tsq THEN ts_rank(tsv_full, tsq)
                 WHEN $2 IS NOT NULL AND content ILIKE '%%'||$2||'%%' THEN 1
                 ELSE 0
               END, 0) AS lex_rank
      FROM knn_ts
    ),
    combined AS (
      SELECT id, content, metadata,
             CASE WHEN lex_rank > 0 THEN $5*sim+$6*lex_rank ELSE sim END AS score
      FROM scored
    )
    SELECT id, content, metadata, score
    FROM combined
    WHERE score >= $8
    ORDER BY score DESC
    LIMIT $3;
  $body$, tbl_full);
  RETURN QUERY EXECUTE sql
    USING query_embedding, query_text, match_count, filter,
          weight_vec, weight_lex, min_cos_sim, min_score, pool_multiplier;
END;
$$;

SET hnsw.ef_search               = 200;
SET ivfflat.probes               = 50;
SET pg_trgm.similarity_threshold = 0.08;
RESET statement_timeout;
