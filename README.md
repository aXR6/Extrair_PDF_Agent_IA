# 🧠 Extração, Chunking e Indexação Inteligente de Documentos PDF/DOCX

## Visão Geral

Este projeto fornece um pipeline completo para processamento de documentos PDF e DOCX, incluindo:

- **Extração de Texto**: múltiplas estratégias (PyPDFLoader, PDFMinerLoader, PDFMiner Low-Level, Unstructured, OCR, PDFPlumber, Tika, PyMuPDF4LLM)
- **Chunking Inteligente**: filtragem de parágrafos, reconhecimento de headings, agrupamento de parágrafos, sliding-window com overlap configurável e fallback para parágrafos longos
- **Embeddings Vetoriais**: suporte a vários modelos (Ollama, Serafim-PT-IR, MPNet, MiniLM) com padding e truncation automáticos
- **Indexação e Busca** híbrida (RAG) com PostgreSQL + pgvector
- **Re-ranking** com Cross-Encoder para maior precisão
- **Monitoramento** via Prometheus (latência, contagem de buscas, tamanho dos resultados)
- **CLI Interativo**: seleção de schema, estratégia de extração, modelo, dimensão, modo verboso e processamento em lote com barra de progresso e estatísticas em tempo real

---

## Funcionalidades

### 1. Extração de Texto

- **Detecção Automática** de PDFs criptografados com fallback para OCR (pytesseract + pdf2image)
- **Estratégias Disponíveis**:
    - PyPDFLoader (LangChain)
    - PDFMinerLoader (LangChain)
    - PDFMiner Low-Level (pdfminer.six)
    - Unstructured (.docx)
    - OCR Hybrid (pytesseract)
    - PDFPlumber
    - Apache Tika
    - PyMuPDF4LLM (Markdown)

### 2. Chunking Inteligente

- **Filtragem de Parágrafos**: remove sumários, índices e trechos curtos (< 50 caracteres)
- **Reconhecimento de Headings**: seções baseadas em headings numéricos
- **Agrupamento de Parágrafos**: utiliza parágrafos inteiros até `max_tokens`
- **Subdivisão de Parágrafos Longos**: sliding-window com overlap (`SLIDING_WINDOW_OVERLAP_RATIO`)
- **Fallback TokenTextSplitter**: para casos de parágrafos acima do limite
- **Expansão de Query**: sinônimos via WordNet em `metadata.__query_expanded`

### 3. Modelos de Embedding & Dimensões

| Opção | Modelo                                                                                       | Dimensão |
|:-----:|:---------------------------------------------------------------------------------------------|:--------:|
| 1     | `mxbai-embed-large` (Ollama API)                                                             | 1024     |
| 2     | `PORTULAN/serafim-900m-portuguese-pt-sentence-encoder-ir` (pt-BR IR)                         | 1536     |
| 3     | `sentence-transformers/all-mpnet-base-v2` (English MPNet)                                    | 768      |
| 4     | `sentence-transformers/all-MiniLM-L6-v2` (MiniLM L6 multilingual)                            | 384      |
| 5     | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (MiniLM L12 multilingual)      | 384      |

> Todos os modelos e dimensões são configuráveis no arquivo `.env`.

### 4. Indexação e Busca

- **Banco:** PostgreSQL + pgvector
- **Tabela:** `public.documents`
    - `id` (BIGSERIAL)
    - `content` (TEXT)
    - `metadata` (JSONB)
    - `embedding` (VECTOR[N])
- **Extensões e Índices:**
    - `vector` (pgvector)
    - IVFFlat / HNSW para vetores
    - `GIN` para `metadata` e `tsv_full`
    - `GIST + pg_trgm` para campos textuais
- **Funções de Busca:**
    - `match_documents_hybrid(...)`
    - `match_documents_precise(...)`

### 5. Re-ranking & Métricas

- **Cross-Encoder:** `ms-marco-MiniLM-L-6-v2` para re-ranking de pares (query, content)
- **Prometheus** (porta 8000):
    - `rag_query_executions_total`
    - `rag_query_duration_seconds`
    - `rag_last_query_result_count`

### 6. CLI Interativo & Estatísticas

- **Menu Principal:**
    1. Selecionar **Schema** (`PG_SCHEMAS` do `.env`)
    2. Selecionar **Estratégia de Extração**
    3. Selecionar **Embedding Model**
    4. Selecionar **Dimensão**
    5. Processar **Arquivo** (tempo e contadores)
    6. Processar **Pasta** (inclui subpastas, contagem total, progresso, sucesso/erros)
    0. Sair
- **Flags:**
    - `--verbose`: logs detalhados
- **Barra de Progresso:** com `tqdm` e `set_postfix` para contadores
- **Resumo Final:** totais de processados, erros e tempo total

---

## Instalação e Uso

**Distro utilizada:** Debian 12 (os comandos podem variar em outras distribuições)

1. **Clone o repositório:**
     ```bash
     git clone https://github.com/seu_usuario/seu_projeto.git
     cd seu_projeto
     ```
2. **Configure o `.env`** conforme exemplo.
3. **Crie e ative o ambiente virtual:**
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```
4. **Instale dependências Python:**
     ```bash
     pip install -r requirements.txt
     ```
5. **Instale dependências de sistema:**
     ```bash
        sudo apt update
        sudo apt install -y \
        poppler-utils   \  # pdftoppm / pdfinfo para pdf2image/pdfminer  
        tesseract-ocr   \  # engine OCR  
        tesseract-ocr-eng tesseract-ocr-por \
        default-jre     \  # para Apache Tika  
        libmagic1       \  # usado por unstructured  
        imagemagick     \  # em alguns setups o python-magic precisa dele  
     ```
6. **Prepare o banco PostgreSQL:**
     - Configure `listen_addresses = '*'` em `postgresql.conf` e ajuste `pg_hba.conf` para acesso remoto.
     - Crie tabelas, extensões (`vector`, `pg_trgm`), índices e funções conforme DDL do projeto.
7. **Execute o CLI:**
     ```bash
     python3 main.py [--verbose]
     ```