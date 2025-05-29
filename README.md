# 🧠 Extração, Chunking e Indexação Inteligente de Documentos PDF/DOCX

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)

## Visão Geral

Pipeline completo para processamento de documentos PDF, DOCX e imagens, incluindo:

- **Extração de Texto:** Diversas estratégias (PyPDFLoader, PDFMinerLoader, PDFMiner Low-Level, Unstructured, OCR para PDF, OCR para Imagens, PDFPlumber, Tika, PyMuPDF4LLM)
- **Chunking Inteligente:** Filtragem de parágrafos, reconhecimento de headings, agrupamento, sliding window com overlap configurável e fallback para parágrafos longos
- **Embeddings Vetoriais:** Suporte a múltiplos modelos (Ollama, Serafim-PT-IR, MPNet, MiniLM) com padding e truncation automáticos
- **Indexação & Busca Híbrida (RAG):** PostgreSQL + pgvector usando tabelas dedicadas por dimensão
- **Re-ranking:** Cross-Encoder (ms-marco) para maior precisão
- **Monitoramento:** Prometheus (latência, contagem de buscas, tamanho dos resultados)
- **CLI Interativo:** Seleção de estratégia, modelo, dimensão, modo verboso, processamento em lote com barra de progresso e estatísticas em tempo real

---

## Funcionalidades

### 1. Extração de Texto

- Detecção automática de PDFs criptografados com fallback para OCR (pytesseract + pdf2image)
- Suporte a OCR direto em imagens (PNG, JPG, JPEG, TIFF, BMP)
- **Estratégias Disponíveis:**
    - PyPDFLoader (LangChain)
    - PDFMinerLoader (LangChain)
    - PDFMiner Low-Level (pdfminer.six)
    - Unstructured (.docx)
    - OCR Hybrid para PDF (pytesseract)
    - ImageOCR (PIL + pytesseract)
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

| Opção | Modelo                                                                                | Dimensão |
|-------|---------------------------------------------------------------------------------------|----------|
| 1     | mxbai-embed-large (Ollama API)                                                        | 1024     |
| 2     | PORTULAN/serafim-900m-portuguese-pt-sentence-encoder-ir (pt-BR IR)                    | 1536     |
| 3     | sentence-transformers/all-mpnet-base-v2 (English MPNet)                               | 768      |
| 4     | sentence-transformers/all-MiniLM-L6-v2 (MiniLM L6 multilingual)                       | 384      |
| 5     | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (MiniLM L12 multilingual) | 384      |

> Todos os modelos e dimensões são configuráveis no arquivo `.env`.

### 4. Indexação e Busca

- **Banco:** PostgreSQL + extensão pgvector
- **Tabelas por dimensão:**
    - `public.documents_384`
    - `public.documents_768`
    - `public.documents_1024`
    - `public.documents_1536`
- **Funções Unificadas:**
    - `public.match_documents_hybrid(query_embedding, query_text, ...)`
    - `public.match_documents_precise(query_embedding, query_text, ...)`
- **Índices e Extensões:**
    - `vector` (pgvector)
    - HNSW / IVFFlat para busca vetorial em cada tabela
    - GIN em `tsv_full` e `metadata`
    - GIN trigram (`gin_trgm_ops`) em `title`, `author`, `type`, `__parent`

### 5. Re-ranking & Métricas

- **Cross-Encoder:** `cross-encoder/ms-marco-MiniLM-L-6-v2` para reranking de pares (query, conteúdo)
- **Prometheus (porta 8000):**
    - `rag_query_executions_total`
    - `rag_query_duration_seconds`
    - `rag_last_query_result_count`

### 6. CLI Interativo & Estatísticas

- **Menu Principal:**
    - Selecionar Estratégia de Extração
    - Selecionar Embedding Model
    - Selecionar Dimensão
    - Processar Arquivo / Pasta (inclui imagens)
    - Sair
- **Flags:**
    - `--verbose`: logs detalhados
- **Progresso:** `tqdm` com `set_postfix` para processados/erros
- **Resumo Final:** totais de processados, erros e tempo total

---

## Requisitos de Sistema

> Testado em **Debian 12** / **Ubuntu 22.04**

- Python 3.8+

### Dependências do Sistema

```bash
sudo apt update
sudo apt install -y \
    poppler-utils \
    mupdf-tools \
    ghostscript \
    qpdf \
    tesseract-ocr \
    tesseract-ocr-eng tesseract-ocr-por \
    libpoppler-cpp-dev pkg-config \
    imagemagick \
    default-jre \
    libmagic1 \
    fontconfig
```

> Para `pdftotext` Python: `pip install pdftotext` após `libpoppler-cpp-dev pkg-config`.

---

## Instalação

1. **Clone o repositório:**

     ```bash
     git clone https://github.com/seu_usuario/seu_projeto.git
     cd seu_projeto
     ```

2. **Crie e ative um virtualenv:**

     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Instale dependências Python:**

     ```bash
     pip install -r requirements.txt
     ```

---

## Exemplo de `.env`

```dotenv
# NVD API Key (para incremental)
NVD_API_KEY=98dbb4f5-7540-4ca1-ae81-ffabf4b076b6

# PostgreSQL Connection
PG_HOST=172.16.187.133
PG_PORT=5432
PG_USER=vector_store
PG_PASSWORD=sua_senha
PG_DATABASE=vector_store

# Modelos & Dimensões
OLLAMA_EMBEDDING_MODEL=mixedbread-ai/mxbai-embed-large-v1
DIM_MXBAI=1024
SERAFIM_EMBEDDING_MODEL=PORTULAN/serafim-900m-portuguese-pt-sentence-encoder-ir
DIM_SERAFIM=1536
MINILM_L6_V2=sentence-transformers/all-MiniLM-L6-v2
DIM_MINILM_L6=384
MINILM_L12_V2=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
DIM_MINIL12=384
MPNET_EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
DIM_MPNET=768

# OCR
OCR_THRESHOLD=100
OCR_LANGUAGES=eng+por

# Chunking
CHUNK_SIZE=1024
CHUNK_OVERLAP=700
SLIDING_WINDOW_OVERLAP_RATIO=0.25
MAX_SEQ_LENGTH=128
SEPARATORS="\n\n|\n|\.|!|\?|;"

# CSV locais (NVD)
CSV_FULL=vulnerabilidades_full.csv
CSV_INCR=vulnerabilidades_incrementais.csv
```

---

## Preparação do Banco PostgreSQL

1. **Instale extensões** dentro do banco:

     ```sql
     CREATE EXTENSION IF NOT EXISTS vector;
     CREATE EXTENSION IF NOT EXISTS pg_trgm;
     ```

2. **Execute o DDL completo** para criar tabelas, dicionários, configuração FTS, triggers e índices conforme as instruções.

---

## Executando o CLI

```bash
python3 main.py [--verbose]
```

---

## Changelog de Exemplo

```yaml
feat: suportar tabelas por dimensão, FTS multilíngue e OCR de imagens
- adicionar ImageOCRStrategy e detectar PNG/JPG/TIFF no extractors
- atualizar is_valid_file e loop de pasta para imagens
- README.md: incluir OCR de imagens nas funcionalidades e CLI
```

---

## Licença

Distribuído sob a licença MIT. Veja [LICENSE](LICENSE) para mais informações.
