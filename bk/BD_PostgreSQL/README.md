# 📚 Estrutura SQL Avançada para Busca Semântica e RAG com PostgreSQL

Este projeto implementa uma estrutura SQL híbrida para busca semântica de documentos, combinando similaridade vetorial (via [pgvector](https://github.com/pgvector/pgvector)) e busca textual (full-text search) com PostgreSQL. Otimizada para aplicações RAG (Retrieval Augmented Generation), integra-se facilmente a pipelines de IA generativa, bots e automações de pesquisa de conteúdo.

---

## Sumário

- [📚 Estrutura SQL Avançada para Busca Semântica e RAG com PostgreSQL](#-estrutura-sql-avançada-para-busca-semântica-e-rag-com-postgresql)
  - [Sumário](#sumário)
  - [Visão Geral](#visão-geral)
  - [Dependências e Extensões](#dependências-e-extensões)
  - [Estrutura da Tabela](#estrutura-da-tabela)
  - [Triggers \& Atualização Automática](#triggers--atualização-automática)
  - [Índices para Performance](#índices-para-performance)
  - [Função Híbrida de Busca `match_documents_hybrid`](#função-híbrida-de-busca-match_documents_hybrid)
  - [Exemplo de Consulta](#exemplo-de-consulta)
  - [Boas Práticas e Observações](#boas-práticas-e-observações)
  - [Testes e Diagnóstico](#testes-e-diagnóstico)
  - [Dicas de Integração](#dicas-de-integração)
  - [Referências](#referências)

---

## Visão Geral

Esta estrutura SQL foi desenvolvida para armazenar documentos segmentados ("chunks") com embeddings vetoriais e suporte a busca textual. O objetivo é permitir consultas híbridas, combinando a precisão da similaridade semântica com a robustez da busca lexical, típico de soluções RAG e chatbots avançados.

---

## Dependências e Extensões

- **PostgreSQL** (recomendado 15+)
- **[pgvector](https://github.com/pgvector/pgvector)** (para armazenamento e busca vetorial)
- **tsvector** (busca textual nativa do PostgreSQL)

Ative as extensões:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

---

## Estrutura da Tabela

A tabela principal é `public.documents`:

```sql
CREATE TABLE IF NOT EXISTS public.documents (
    id BIGSERIAL PRIMARY KEY,
    content TEXT NOT NULL,         -- Conteúdo do chunk/documento
    metadata JSONB NOT NULL,       -- Metadados (autor, título, origem, chunk_index etc)
    embedding VECTOR(1024) NOT NULL, -- Embedding vetorial (1024 dims)
    tsv_full TSVECTOR              -- Campo para busca textual otimizada
);
```

**Observações:**
- Embeddings: vetores de dimensão 1024 (compatível com OpenAI, Ollama, etc.).
- Metadados flexíveis em formato JSONB.
- Campo `tsv_full` é mantido automaticamente via trigger.

---

## Triggers & Atualização Automática

Sempre que um registro é inserido/atualizado, o campo `tsv_full` é atualizado com a concatenação ponderada do texto e dos metadados:

```sql
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
```

- **Ponderação:** Conteúdo recebe peso 'A', metadados peso 'B'.

---

## Índices para Performance

- **IVFFlat:** Otimiza a busca vetorial (KNN). `lists = 500` para grandes volumes.
- **HNSW:** Alternativa para workloads específicos ou busca rápida em larga escala.
- **GIN em tsv_full:** Busca textual rápida, inclusive para filtros dinâmicos.

```sql
CREATE INDEX IF NOT EXISTS idx_documents_embedding_ivfflat
    ON public.documents USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 500);

CREATE INDEX IF NOT EXISTS idx_documents_embedding_hnsw
    ON public.documents USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 200);

CREATE INDEX IF NOT EXISTS idx_documents_tsv_full
    ON public.documents USING gin(tsv_full);
```

**Tuning de consultas:**

```sql
SET ivfflat.probes = 50;
SET hnsw.ef_search = 100;
```

---

## Função Híbrida de Busca `match_documents_hybrid`

Realiza a busca combinada vetorial + lexical (busca semântica + full-text):

```sql
CREATE OR REPLACE FUNCTION public.match_documents_hybrid(
    query_embedding VECTOR(1024),
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
            knn_ts.parent, 
            knn_ts.id, 
            knn_ts.content, 
            knn_ts.metadata, 
            1 - knn_ts.dist AS sim,
            COALESCE(
                CASE WHEN query_tsq IS NOT NULL AND knn_ts.tsv_full @@ query_tsq
                    THEN ts_rank(knn_ts.tsv_full, query_tsq)
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
```

**Explicação:**
- Pré-seleção: Seleciona um pool ampliado de chunks mais próximos vetorialmente (KNN).
- Busca textual: Calcula `ts_rank` apenas nesse pool (eficiência).
- Score final: Combina similaridade vetorial e ranking textual, ponderados por `weight_vec` e `weight_lex`.
- Agrupamento por "parent": Retorna apenas os chunks do documento-pai mais relevante.
- Threshold: Filtra resultados com score abaixo do mínimo.

---

## Exemplo de Consulta

```sql
SELECT * FROM public.match_documents_hybrid(
    $1::vector(1024),   -- embedding de consulta
    $2::text,           -- texto de consulta (query)
    $3::int,            -- match_count (máximo de resultados)
    $4::jsonb,          -- filtro por metadados (ex: '{"author":"Nome"}')
    $5::float,          -- peso da similaridade vetorial (0.0 a 1.0)
    $6::float,          -- peso da busca lexical (0.0 a 1.0)
    $7::float,          -- threshold mínimo de score
    $8::int             -- pool_multiplier (tamanho do pool KNN, padrão 10)
);
```

> **Obs:** Perfeito para integração com n8n, Node.js, Python, etc.  
> Passe o vetor de embedding e os parâmetros conforme seu pipeline.

---

## Boas Práticas e Observações

- Sempre qualifique os campos (ex: `documents.metadata`) em queries e CTEs para evitar ambiguidade.
- Ajuste `lists`, `m`, `ef_construction` e `probes` dos índices vetoriais conforme seu volume de dados.
- Filtros avançados podem ser aplicados facilmente via campo `metadata` (JSONB).
- Chunks devem ser construídos de modo semântico para maximizar precisão na busca.

---

## Testes e Diagnóstico

Valide após carga massiva:

```sql
SELECT count(*) FROM documents WHERE tsv_full IS NULL;
```

Caso >0, force update:

```sql
UPDATE documents SET content = content; -- Dispara trigger para atualizar tsv_full
```

Reindexe após grande atualização:

```sql
REINDEX INDEX idx_documents_tsv_full;
```

> Mensagens “palavra muito longa para ser indexada” indicam que termos excederam o limite do PostgreSQL (limite default: 2k chars).

---

## Dicas de Integração

- **RAG e Chatbots:** Ideal para pipelines de busca de contexto em IA generativa.
- **n8n:** Integração via node Postgres, passando parâmetros de embedding, texto e filtro.
- **Python/Node:** Use drivers padrão (`psycopg2`, `node-postgres`) e envie parâmetros na ordem correta.

---

## Referências

- [pgvector](https://github.com/pgvector/pgvector)
- [Full Text Search - PostgreSQL](https://www.postgresql.org/docs/current/textsearch.html)
- [RAG with Postgres & LangChain](https://python.langchain.com/docs/integrations/vectorstores/pgvector)
