# 🧠 Extração, Chunking e Indexação Inteligente de Documentos PDF/DOCX

## Visão Geral

Este projeto oferece um pipeline completo para processamento de documentos PDF e DOCX, incluindo:

- **Extração de texto** com múltiplas estratégias (PyPDFLoader, PDFMinerLoader, PDFMiner Low-Level, Unstructured, OCR, PDFPlumber, Tika, PyMuPDF4LLM)
- **Chunking inteligente**: filtragem de parágrafos, reconhecimento de headings, agrupamento de parágrafos inteiros, sliding window com overlap configurável e fallback para parágrafos longos
- **Embeddings vetoriais**: suporte a múltiplos modelos (Ollama, Serafim-PT-IR, MPNet, MiniLM) com padding e truncation automáticos
- **Indexação e busca híbrida** (RAG) com PostgreSQL + pgvector
- **Re-ranking** com Cross-Encoder (ms-marco-MiniLM-L-6-v2) para maior precisão
- **Monitoramento** via Prometheus (latência, contagem de buscas, tamanho dos resultados)
- **CLI interativo**: seleção de schema, estratégia de extração, modelo, dimensão, modo verboso e processamento em lote com barra de progresso e estatísticas em tempo real

---

## Funcionalidades (Distro utilizada: Debian 12)

### Extração de Texto

- **Detecção automática** de PDFs criptografados com fallback para OCR (pytesseract + pdf2image)
- **Estratégias disponíveis**:
    - PyPDFLoader (LangChain)
    - PDFMinerLoader (LangChain)
    - PDFMiner Low-Level (pdfminer.six)
    - Unstructured (.docx)
    - OCR Hybrid (pytesseract)
    - PDFPlumber
    - Apache Tika
    - PyMuPDF4LLM (Markdown)

### Chunking Inteligente

- **Filtragem de parágrafos**: remove sumários, índices e trechos curtos (< 50 caracteres)
- **Reconhecimento de headings**: seções semânticas baseadas em headings numéricos
- **Agrupamento de parágrafos**: utiliza parágrafos inteiros até `max_tokens`
- **Subdivisão de parágrafos longos**: sliding window com overlap (`SLIDING_WINDOW_OVERLAP_RATIO`)
- **Fallback TokenTextSplitter**: para casos excepcionais de parágrafos maiores que o limite
- **Expansão de query**: sinônimos via WordNet em `metadata.__query_expanded`

### Modelos de Embedding e Dimensões

| Opção | Modelo                                                                                       | Dimensão |
|:-----:|:---------------------------------------------------------------------------------------------|:--------:|
| 1     | `mxbai-embed-large` (Ollama API)                                                             | 1024     |
| 2     | `PORTULAN/serafim-900m-portuguese-pt-sentence-encoder-ir` (pt-BR IR)                         | 1536     |
| 3     | `sentence-transformers/all-mpnet-base-v2` (English MPNet)                                    | 768      |
| 4     | `sentence-transformers/all-MiniLM-L6-v2` (MiniLM L6 multilingual)                            | 384      |
| 5     | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (MiniLM L12 multilingual)      | 384      |

> Todos os modelos e dimensões são configurados no arquivo `.env`.

### Indexação e Busca

- **Tabela `public.documents`**:
    - `id` (BIGSERIAL)
    - `content` (TEXT)
    - `metadata` (JSONB)
    - `embedding` (VECTOR[N])
- **Extensões e índices**:
    - `vector` (pgvector)
    - IVFFlat / HNSW para vetores
    - `GIN` para `metadata` e `tsv_full`
    - `GIST + pg_trgm` para campos textuais críticos
- **Funções de busca**:
    - `match_documents_hybrid(...)`
    - `match_documents_precise(...)`

### Re-ranking e Métricas

- **Cross-Encoder**: `ms-marco-MiniLM-L-6-v2` para re-ranking de pares (query, content)
- **Prometheus** (porta 8000):
    - `rag_query_executions_total`
    - `rag_query_duration_seconds`
    - `rag_last_query_result_count`

### CLI Interativo e Estatísticas

- **Menu principal**:
    1. Selecionar schema (`PG_SCHEMAS` do `.env`)
    2. Selecionar estratégia de extração
    3. Selecionar embedding model
    4. Selecionar dimensão
    5. Processar arquivo (tempo e contadores)
    6. Processar pasta (inclui subpastas, contagem total, progresso, sucesso/erros)
    0. Sair
- **Flags**:
    - `--verbose`: habilita logs detalhados
- **Barra de progresso** com `tqdm` e `set_postfix` para contadores
- **Resumo final**: totais de processados, erros e tempo total

---

## Instalação e Uso

1. **Clone o repositório**:
     ```bash
     git clone https://github.com/seu_usuario/seu_projeto.git
     cd seu_projeto
     ```
2. **Configure o `.env`** conforme exemplo abaixo
3. **Instale as dependências**:
     ```bash
     pip install -r requirements.txt
     ```
4. **Prepare o banco PostgreSQL** (extensões, tabela, índices, funções)
5. **Execute o CLI**:
     ```bash
     python3 main.py [--verbose]
     ```

---

## Exemplo de `.env`

```dotenv
# NVD
NVD_API_KEY=98dbb4f5-7540-4ca1-ae81-ffabf4b076b6

# PostgreSQL
PG_HOST=192.168.3.32
PG_PORT=5432
PG_SCHEMAS=vector_1024,vector_384,vector_768,vector_1536
PG_SCHEMA_DEFAULT=vector_384
PG_USER=vector_store
PG_PASSWORD=senha

# Modelos
OLLAMA_EMBEDDING_MODEL=mxbai-embed-large
SERAFIM_EMBEDDING_MODEL=PORTULAN/serafim-900m-portuguese-pt-sentence-encoder-ir
MINILM_L6_V2=sentence-transformers/all-MiniLM-L6-v2
MINILM_L12_V2=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
MPNET_EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
SBERT_MODEL_NAME=${OLLAMA_EMBEDDING_MODEL}

# Dimensões
DIM_MXBAI=1024
DIM_SERAFIM=1536
DIM_MINILM_L6=384
DIM_MINIL12=384
DIM_MPNET=768

# OCR
OCR_THRESHOLD=100

# Chunking
CHUNK_SIZE=1024
CHUNK_OVERLAP=700
SLIDING_WINDOW_OVERLAP_RATIO=0.25
MAX_SEQ_LENGTH=128
SEPARATORS="\n\n|\n|.|!|?|;"

# CSV NVD
CSV_FULL=vulnerabilidades_full.csv
CSV_INCR=vulnerabilidades_incrementais.csv
```

---

## Exemplo de Commit

```text
feat: atualiza README para versão estável 1.0.0

- Documenta chunking inteligente e CLI com progresso
- Inclui seleção de schema e flag `--verbose`
- Atualiza modelo Serafim-PT-IR no `.env`
```
