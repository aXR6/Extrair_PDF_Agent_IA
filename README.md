# 🧠 Extração, Chunking e Indexação Inteligente de Documentos PDF/DOCX

## Visão Geral

Este projeto oferece um pipeline completo para processamento de documentos PDF e DOCX, incluindo:

- **Extração de Texto:** Diversas estratégias (PyPDFLoader, PDFMinerLoader, PDFMiner Low-Level, Unstructured, OCR, PDFPlumber, Tika, PyMuPDF4LLM)
- **Chunking Inteligente:** Filtragem de parágrafos, reconhecimento de headings, agrupamento, sliding window com overlap configurável e fallback para parágrafos longos
- **Embeddings Vetoriais:** Suporte a múltiplos modelos (Ollama, Serafim-PT-IR, MPNet, MiniLM) com padding e truncation automáticos
- **Indexação & Busca Híbrida (RAG):** PostgreSQL + pgvector
- **Re-ranking:** Cross-Encoder (ms-marco) para maior precisão
- **Monitoramento:** Prometheus (latência, contagem de buscas, tamanho dos resultados)
- **CLI Interativo:** Seleção de schema, estratégia, modelo, dimensão, modo verboso, processamento em lote com barra de progresso e estatísticas em tempo real

---

## Funcionalidades

### 1. Extração de Texto

- Detecção automática de PDFs criptografados com fallback para OCR (pytesseract + pdf2image)
- **Estratégias Disponíveis:**
    - PyPDFLoader (LangChain)
    - PDFMinerLoader (LangChain)
    - PDFMiner Low-Level (pdfminer.six)
    - Unstructured (.docx)
    - OCR Hybrid (pytesseract)
    - PDFPlumber
    - Apache Tika
    - PyMuPDF4LLM (Markdown)

### 2. Chunking Inteligente

- Filtragem de parágrafos: remove sumários, índices e trechos muito curtos (< 50 caracteres)
- Reconhecimento de headings: seções baseadas em padrões numéricos (ex: 1.2 Título)
- Agrupamento de parágrafos até `max_tokens`
- Subdivisão de parágrafos longos: sliding window com overlap (`SLIDING_WINDOW_OVERLAP_RATIO`)
- Fallback `TokenTextSplitter` para casos excepcionais
- Expansão de Query: sinônimos via WordNet em `metadata.__query_expanded`

### 3. Modelos de Embedding & Dimensões

| Opção | Modelo                                                                 | Dimensão |
|-------|------------------------------------------------------------------------|----------|
| 1     | mxbai-embed-large (Ollama API)                                         | 1024     |
| 2     | PORTULAN/serafim-900m-portuguese-pt-sentence-encoder-ir (pt-BR IR)     | 1536     |
| 3     | sentence-transformers/all-mpnet-base-v2 (English MPNet)                | 768      |
| 4     | sentence-transformers/all-MiniLM-L6-v2 (MiniLM L6 multilingual)        | 384      |
| 5     | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (MiniLM L12 multilingual) | 384      |

> Todos os modelos e dimensões são configuráveis no arquivo `.env`.

### 4. Indexação e Busca

- **Banco:** PostgreSQL + extensão pgvector
- **Tabela:** `public.documents`
    - `id` (BIGSERIAL)
    - `content` (TEXT)
    - `metadata` (JSONB)
    - `embedding` (VECTOR[N])
- **Índices e Extensões:**
    - `vector` (pgvector)
    - IVFFlat / HNSW para vetores
    - GIN em metadata e tsv_full
    - GIST + pg_trgm para campos textuais
- **Stored Procedures / Funções:**
    - `match_documents_hybrid(query_embedding, query_text, …)`
    - `match_documents_precise(query_embedding, query_text, …)`

### 5. Re-ranking & Métricas

- **Cross-Encoder:** `cross-encoder/ms-marco-MiniLM-L-6-v2` para reranking de pares (query, conteúdo)
- **Prometheus (porta 8000):**
    - `rag_query_executions_total`
    - `rag_query_duration_seconds`
    - `rag_last_query_result_count`

### 6. CLI Interativo & Estatísticas

- **Menu Principal:**
    - Selecionar Schema (`PG_SCHEMAS` no `.env`)
    - Selecionar Estratégia de Extração
    - Selecionar Embedding Model
    - Selecionar Dimensão
    - Processar Arquivo (tempo e contadores)
    - Processar Pasta (inclui subpastas, contagem, progresso, sucesso/erros)
    - Sair
- **Flags:**
    - `--verbose`: logs detalhados
- **Progresso:** `tqdm` com `set_postfix` para P/E
- **Resumo Final:** totais de processados, erros e tempo total

---

## Requisitos de Sistema

> Testado em **Debian 12** / **Ubuntu 22.04**

- Python 3.8+ instalado

### Dependências do Sistema

```bash
sudo apt update
sudo apt install -y \
        poppler-utils \         # pdftoppm, pdfinfo (poppler-utils)
        mupdf-tools \           # mutool (MuPDF) para 'mutool clean'
        ghostscript \           # gs para fallback Ghostscript
        qpdf \                  # pikepdf/QPDF engine
        tesseract-ocr \
        tesseract-ocr-eng tesseract-ocr-por \
        libpoppler-cpp-dev pkg-config \ # para compilar 'pdftotext' Python
        imagemagick \           # requerido por alguns backends de imagem
        default-jre \           # para Apache Tika
        libmagic1 \             # para python-magic/unstructured
        fontconfig              # renderização de fontes (pdf2image)
```

> **Observação:**  
> Para usar a lib Python `pdftotext`, instale também `libpoppler-cpp-dev` e `pkg-config` antes de `pip install pdftotext`.  
> O pacote `pikepdf` não existe no APT; instale via `pip install pikepdf`.

---

## Instalação Python

1. **Clone o repositório e entre na pasta:**

        ```bash
        git clone https://github.com/seu_usuario/seu_projeto.git
        cd seu_projeto
        ```

2. **Configure o `.env` (veja seção abaixo)**

3. **Crie e ative um virtualenv:**

        ```bash
        python3 -m venv .venv
        source .venv/bin/activate
        ```

4. **Instale as dependências Python:**

        ```bash
        pip install -r requirements.txt
        ```

---

## Exemplo de `.env`

```dotenv
# — NVD API Key (para incremental)
NVD_API_KEY=98dbb4f5-7540-4ca1-ae81-ffabf4b076b6

# — PostgreSQL Connection
PG_HOST=172.16.187.133
PG_PORT=5432
PG_USER=vector_store
PG_PASSWORD=senha_secreta
PG_SCHEMAS=vector_1024,vector_384,vector_768,vector_1536
PG_SCHEMA_DEFAULT=vector_384

# — Modelos & Dimensões
OLLAMA_EMBEDDING_MODEL=mxbai-embed-large
DIM_MXBAI=1024

SERAFIM_EMBEDDING_MODEL=PORTULAN/serafim-900m-portuguese-pt-sentence-encoder-ir
DIM_SERAFIM=1536

MINILM_L6_V2=sentence-transformers/all-MiniLM-L6-v2
DIM_MINILM_L6=384

MINILM_L12_V2=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
DIM_MINIL12=384

MPNET_EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
DIM_MPNET=768

# — SBERT (unifica chunking & embedding)
SBERT_MODEL_NAME=${OLLAMA_EMBEDDING_MODEL}

# — OCR
OCR_THRESHOLD=100
OCR_LANGUAGES=eng+por

# — Chunking
CHUNK_SIZE=1024
CHUNK_OVERLAP=700
SLIDING_WINDOW_OVERLAP_RATIO=0.25
MAX_SEQ_LENGTH=128
SEPARATORS="\n\n|\n|\.|!|\?;"

# — CSV locais (NVD)
CSV_FULL=vulnerabilidades_full.csv
CSV_INCR=vulnerabilidades_incrementais.csv
```

---

## Preparação do Banco PostgreSQL

1. **Habilite extensões no seu `postgresql.conf`:**

        ```conf
        shared_preload_libraries = 'vector'
        ```

2. **Em `pg_hba.conf`, permita acesso remoto (se necessário).**

3. **Dentro do banco/schema escolhido, execute o DDL:**

        ```sql
        CREATE EXTENSION IF NOT EXISTS vector;
        CREATE EXTENSION IF NOT EXISTS pg_trgm;

        CREATE TABLE public.documents (
                id BIGSERIAL PRIMARY KEY,
                content TEXT NOT NULL,
                metadata JSONB NOT NULL,
                embedding VECTOR(<DIM>) NOT NULL
        );
        -- índice HNSW / IVFFlat:
        CREATE INDEX ON public.documents USING ivfflat (embedding vector_l2_ops) WITH (lists = 100);
        -- índice GIN para metadata:
        CREATE INDEX ON public.documents USING gin (metadata);
        -- índice GIST + pg_trgm (ex.: campo 'info.title'):
        CREATE INDEX ON public.documents USING gist ((metadata->>'title') gist_trgm_ops);
        ```

---

## Executando o CLI

```bash
python3 main.py [--verbose]
```

- Na primeira execução, o modelo SBERT será baixado pelo `sentence-transformers`.
- Para mudar de schema, estratégia ou modelo, basta seguir o menu interativo.

---

## Exemplo de Commit

```yaml
feat: instalar poppler-utils, mupdf-tools, ghostscript e pdftotext
- apt: poppler-utils, mupdf-tools, ghostscript, qpdf, libpoppler-cpp-dev, pkg-config, default-jre, libmagic1
- requirements.txt: adiciona pikepdf e pdftotext
- extractors.py: fallback pdftotext (poppler ou pip)
- utils.py: inclui mutool clean
- README.md: atualiza instruções de instalação
```
