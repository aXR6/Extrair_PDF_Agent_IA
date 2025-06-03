# 📚 Estrutura SQL Avançada para Busca Semântica e RAG com PostgreSQL

Este módulo define a **estrutura de banco** para indexação e busca híbrida (RAG) em múltiplas dimensões de embedding, utilizando **PostgreSQL 17**, extensão **pgvector** e **Full-Text Search** multilíngue.

---

## Sumário

- [📚 Estrutura SQL Avançada para Busca Semântica e RAG com PostgreSQL](#-estrutura-sql-avançada-para-busca-semântica-e-rag-com-postgresql)
  - [Sumário](#sumário)
  - [Visão Geral](#visão-geral)
  - [Dependências e Extensões](#dependências-e-extensões)
  - [Configuração FTS Multilíngue](#configuração-fts-multilíngue)
  - [Tabelas por Dimensão](#tabelas-por-dimensão)
  - [Triggers \& Atualização Automática](#triggers--atualização-automática)
  - [Índices para Performance](#índices-para-performance)
  - [Funções Unificadas de Busca](#funções-unificadas-de-busca)
  - [Exemplo de Uso](#exemplo-de-uso)
  - [Boas Práticas](#boas-práticas)
  - [Referências](#referências)

---

## Visão Geral

Armazena **chunks** de documentos com embeddings e tsvectors, permitindo consultas híbridas que combinam similaridade vetorial e busca textual. Suporta **4 dimensões** de vetor (384, 768, 1024 e 1536) em **tabelas independentes**, facilitando escalabilidade e manutenção.

---

## Dependências e Extensões

- **PostgreSQL 17+**
- **pgvector** (vetores e índices vetoriais)
- **pg_trgm** (trigramas para filtros fuzzy)

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
```

---

## Configuração FTS Multilíngue

Cria configuração que combina stemmers para **português** e **inglês**:

```sql
-- remove se existente para redefinir
DROP TEXT SEARCH CONFIGURATION IF EXISTS public.pt_en;
CREATE TEXT SEARCH CONFIGURATION public.pt_en (COPY = pg_catalog.simple);
ALTER TEXT SEARCH CONFIGURATION public.pt_en
  ALTER MAPPING FOR asciiword, asciihword, hword_asciipart
  WITH portuguese, english, simple;
```

---

## Tabelas por Dimensão

Cada dimensão de embedding tem sua própria tabela no schema `public`:

```sql
-- Exemplo para dimensão 384
CREATE TABLE IF NOT EXISTS public.documents_384 (
  id         BIGSERIAL    PRIMARY KEY,
  content    TEXT         NOT NULL,
  metadata   JSONB        NOT NULL,
  embedding  VECTOR(384)  NOT NULL,
  tsv_full   TSVECTOR,
  created_at TIMESTAMPTZ  DEFAULT now()
);
```

As tabelas **768**, **1024** e **1536** seguem o mesmo modelo (clonando `documents_384` e alterando apenas `VECTOR(n)`).

---

## Triggers & Atualização Automática

Mantém `tsv_full` sempre atualizado:

```sql
CREATE OR REPLACE FUNCTION public.update_tsv_full() RETURNS trigger AS $$
BEGIN
  NEW.tsv_full :=
    setweight(to_tsvector('public.pt_en', coalesce(NEW.content, '')), 'A') ||
    setweight(to_tsvector('public.pt_en', coalesce(NEW.metadata::text, '')), 'B');
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Em cada tabela:
DROP TRIGGER IF EXISTS tsv_full_trigger ON public.documents_384;
CREATE TRIGGER tsv_full_trigger
  BEFORE INSERT OR UPDATE ON public.documents_384
  FOR EACH ROW EXECUTE FUNCTION public.update_tsv_full();
```

---

## Índices para Performance

- **Vetoriais (pgvector):** HNSW e IVFFlat
- **GIN** em `tsv_full` e `metadata`
- **GIN trigram** em campos textuais (`title`, `author`, `type`, `__parent`)

```sql
-- Exemplo HNSW + IVFFlat
CREATE INDEX ON public.documents_384 USING hnsw (embedding vector_cosine_ops) WITH (m=16, ef_construction=200);
CREATE INDEX ON public.documents_384 USING ivfflat (embedding vector_cosine_ops) WITH (lists=400);

-- GIN full-text
CREATE INDEX ON public.documents_384 USING gin(tsv_full);
CREATE INDEX ON public.documents_384 USING gin(metadata);
-- trigram ops
CREATE INDEX ON public.documents_384 USING gin((metadata->>'title') gin_trgm_ops);
CREATE INDEX ON public.documents_384 USING gin((metadata->>'__parent') gin_trgm_ops);
```

Ajuste parâmetros de busca:
```sql
SET hnsw.ef_search = 200;
SET ivfflat.probes = 50;
SET pg_trgm.similarity_threshold = 0.08;
```

---

## Funções Unificadas de Busca

As funções detectam a dimensão do vetor e direcionam para a tabela correta:

```sql
CREATE OR REPLACE FUNCTION public.match_documents_hybrid(
  query_embedding vector,
  query_text      text      DEFAULT NULL,
  match_count     int       DEFAULT 5,
  filter          jsonb     DEFAULT '{}'::jsonb,
  weight_vec      float     DEFAULT 0.6,
  weight_lex      float     DEFAULT 0.4,
  min_score       float     DEFAULT 0.1,
  pool_multiplier int       DEFAULT 10
) RETURNS TABLE (id bigint, content text, metadata jsonb, score float) AS $$
DECLARE
  dim int := vector_dims(query_embedding);
  tbl text := format('public.documents_%s', dim);
  sql text;
BEGIN
  IF dim NOT IN (384,768,1024,1536) THEN
    RAISE EXCEPTION 'Dim % não suportada', dim;
  END IF;
  sql := format($f$ /* CTE híbrida semelhante ao single-table */ $f$, tbl);
  RETURN QUERY EXECUTE sql USING query_embedding, query_text, match_count,
    filter, weight_vec, weight_lex, min_score, pool_multiplier;
END;
$$ LANGUAGE plpgsql;
```

Também há **`match_documents_precise`** com filtro mínimo de cosseno.

---

## Exemplo de Uso

```sql
-- embedding = vetor de 1536 dims, query_text = '...', filter = '{}'::jsonb
SELECT *
FROM public.match_documents_hybrid(
  $1::vector(1536),
  $2::text,
  5,
  $3::jsonb
);
```

---

## Boas Práticas

- **Qualifique** sempre tabelas e colunas em queries.
- **Reindexe** após grandes cargas: `REINDEX INDEX idx_documents_384_tsv_full;`.
- **Ajuste** `lists`, `m`, `ef_construction`, `probes` conforme dados.
- Use filtros JSONB para refinamentos (ex: `{"author":"X"}`).

---

## Referências

- [pgvector](https://github.com/pgvector/pgvector)
- [Full Text Search - PostgreSQL](https://www.postgresql.org/docs/current/textsearch.html)
- [RAG com Postgres & LangChain](https://python.langchain.com/docs/integrations/vectorstores/pgvector)
