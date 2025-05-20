-- 1) Extensões Necessárias
CREATE EXTENSION IF NOT EXISTS vector;       -- pgvector para embeddings
CREATE EXTENSION IF NOT EXISTS pg_trgm;      -- trigramas para fuzzy text filtering

-- 2) Tabela de Documentos (mantida como está)
CREATE TABLE IF NOT EXISTS public.documents (
  id        BIGSERIAL PRIMARY KEY,
  content   TEXT         NOT NULL,
  metadata  JSONB        NOT NULL,
  embedding VECTOR(384)  NOT NULL,
  tsv_full  TSVECTOR
);

-- 3) Função e Trigger de tsv_full (full-text híbrido)
CREATE OR REPLACE FUNCTION public.update_tsv_full() RETURNS trigger AS $$
BEGIN
  NEW.tsv_full :=
    setweight(to_tsvector('simple', coalesce(NEW.content, '')), 'A') ||
    setweight(to_tsvector('simple', coalesce(NEW.metadata::text, '')), 'B');
  RETURN NEW;
END
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS tsv_full_trigger ON public.documents;
CREATE TRIGGER tsv_full_trigger
  BEFORE INSERT OR UPDATE ON public.documents
  FOR EACH ROW EXECUTE FUNCTION public.update_tsv_full();

-- 4) Índice Vetorial HNSW (padrão primário para RAG)
CREATE INDEX IF NOT EXISTS idx_documents_embedding_hnsw
  ON public.documents
  USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 200);

-- 5) Índice Vetorial IVFFlat (opcional como fallback)
CREATE INDEX IF NOT EXISTS idx_documents_embedding_ivfflat
  ON public.documents
  USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 200);

-- 6) Índice GIN Global para JSONB Metadata
CREATE INDEX IF NOT EXISTS idx_documents_metadata_gin
  ON public.documents
  USING gin (metadata);

-- 7) Índices GiST+pg_trgm em Campos de Metadata Críticos
CREATE INDEX IF NOT EXISTS idx_documents_title_trgm
  ON public.documents
  USING gist ((metadata->>'title') gist_trgm_ops);

CREATE INDEX IF NOT EXISTS idx_documents_author_trgm
  ON public.documents
  USING gist ((metadata->>'author') gist_trgm_ops);

CREATE INDEX IF NOT EXISTS idx_documents_parent_trgm
  ON public.documents
  USING gist ((metadata->>'__parent') gist_trgm_ops);

-- 8) Índice Full-Text Híbrido em tsv_full
CREATE INDEX IF NOT EXISTS idx_documents_tsv_full
  ON public.documents
  USING gin (tsv_full);

-- 9) Parâmetros de Sessão Recomendados para Consulta RAG
-- Em cada sessão de consulta (por exemplo, no n8n ou aplicação):
SET hnsw.ef_search = 200;                     -- aumenta recall HNSW :contentReference[oaicite:7]{index=7}
SET ivfflat.probes   = 50;                    -- ajusta balanceamento IVFFlat :contentReference[oaicite:8]{index=8}
SET pg_trgm.similarity_threshold = 0.1;       -- captura fuzzy matches mais amplos :contentReference[oaicite:9]{index=9}

-- 10) Exemplo de Função de Busca Híbrida (mantida ou levemente ajustada)
-- (já fornecida em sua estrutura; garante combinação vetorial+lexical para RAG)