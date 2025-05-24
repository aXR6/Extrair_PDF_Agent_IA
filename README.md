# 🧠 Extração, Chunking e Indexação Inteligente de Documentos PDF/DOCX

## Visão Geral

Este projeto oferece um pipeline completo para processamento de documentos PDF e DOCX, incluindo:

- **Extração de Texto** com múltiplas estratégias (PyPDF2, PDFMiner, OCR, Tika, Unstructured, PDFPlumber, PyMuPDF4LLM)
- **Chunking Inteligente**: filtros de parágrafos, enriquecimento semântico (sumarização, NER, paráfrase, expansão de queries) e sliding-window
- **Embeddings Vetoriais**: suporte a múltiplos modelos (Ollama, Serafim-PT, MPNet-EN, MiniLM)
- **Indexação e Busca** híbrida (RAG) no PostgreSQL/pgvector e MongoDB/GridFS
- **Re-ranking** com Cross-Encoder (ms-marco) para maior precisão
- **Monitoramento** via Prometheus (latência, contagem de buscas, tamanho dos resultados)
- **CLI Interativo**: seleção de estratégia, banco, schema, modelo, dimensão e batch-processing recursivo

O CLI processa automaticamente todos os subdiretórios e “pastas irmãs” do caminho raiz indicado, com barra de progresso.

---

## Funcionalidades

### 1. Extração de Texto

- **Híbrido automático**: detecta PDFs criptografados e faz fallback OCR (pytesseract) se necessário
- **Estratégias disponíveis**:
    - PyPDFLoader (LangChain)
    - PDFMinerLoader (LangChain)
    - PDFMiner Low-Level (pdfminer.six)
    - Unstructured (.docx)
    - OCR Strategy (pytesseract + pdf2image)
    - PDFPlumber
    - Apache Tika
    - PyMuPDF4LLM → Markdown

### 2. Chunking Inteligente

1. **Filtragem de parágrafos**: remove sumários, índices e trechos curtos (< 50 caracteres)
2. **Divisão hierárquica**: reconhece headings numéricos (`1.2 Seção`) e quebra em seções semânticas
3. **Enriquecimento de conteúdo**:
     - **Sumarização** (`sshleifer/distilbart-cnn-12-6`)
     - **NER** (`dbmdz/bert-large-cased-finetuned-conll03-english`)
     - **Paráfrase** (`t5-small`)
4. **Sliding-Window**: sobreposição percentual configurável (`SLIDING_WINDOW_OVERLAP_RATIO`)
5. **TokenTextSplitter**: separadores customizados para textos longos
6. **Padding / Truncamento**: embeddings ajustados para dimensão estável
7. **Expansão de Query**: sinônimos via WordNet em `metadata.__query_expanded`

### 3. Modelos de Embedding & Dimensões

Disponíveis no menu CLI:

| Opção | Modelo                                                                                   | Dimensão |
|:-----:|:-----------------------------------------------------------------------------------------|:--------:|
| 1     | `mxbai-embed-large` (Ollama API)                                                         | 1024     |
| 2     | `PORTULAN/serafim-900m-portuguese-pt-sentence-encoder` (pt-BR)                           | 1536     |
| 3     | `sentence-transformers/all-mpnet-base-v2` (English MPNet)                                | 768      |
| 4     | `sentence-transformers/all-MiniLM-L6-v2` (MiniLM L6 multilingual)                        | 384      |
| 5     | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (MiniLM L12 multilingual)  | 384      |

> O arquivo `.env` contém variáveis para cada modelo e dimensão (ver seção “Configuração”).

### 4. Indexação e Busca

#### MongoDB

- Documentos chunked + metadados na collection definida
- Binários em GridFS

#### PostgreSQL + pgvector

- Tabela `public.documents`:
    - `id` (BIGSERIAL)
    - `content` (TEXT)
    - `metadata` (JSONB)
    - `embedding` (VECTOR(N)) — N = dimensão do modelo escolhido
    - `tsv_full` (TSVECTOR)
- Triggers e funções para manter `tsv_full` atualizado automaticamente
- Índices:
    - **HNSW** / **IVFFlat** para vetores
    - **GIN** em `tsv_full`
    - **GIST+pg_trgm** para campos críticos de metadata

#### Funções de Busca

- `match_documents_hybrid(query_embedding, query_text, …)`
- `match_documents_precise(query_embedding, query_text, …)`

### 5. Re-ranking & Métricas

- **Cross-Encoder** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) para reranking de pares (query, content)
- **Prometheus**:
    - Métricas expostas em `/metrics` (porta 8000):
        - `rag_query_executions_total`
        - `rag_query_duration_seconds`
        - `rag_last_query_result_count`

### 6. CLI Interativo & Batch Processing

- **Menu Principal**:
    1. Selecionar estratégia de extração
    2. Selecionar SGBD (MongoDB ou PostgreSQL)
    3. (se PostgreSQL) Selecionar schema (`vector_1024`, `vector_384`, `vector_1536`, `vector_768`)
    4. Processar **arquivo**
    5. Processar **pasta** (inclui subpastas + “pastas irmãs”)
    6. Selecionar modelo de embedding
    7. Selecionar dimensão
    0. Sair
- **Progress Bar** via `tqdm`
- **Arquivos finalizados** movidos para subpasta `processed`

---

## Como Usar

1. **Clone o repositório**
     ```bash
     git clone https://github.com/seu_usuario/seu_projeto.git
     cd seu_projeto
     ```
2. **Ajuste o `.env`** (veja seção abaixo)
3. **Instale as dependências**
     ```bash
     pip install -r requirements.txt
     ```
4. **Aplique o DDL no PostgreSQL** para criar extensões, tabela, triggers e funções
5. **Execute o CLI**
     ```bash
     python3 main.py
     ```

--- 

## Exemplo de `.env`

```dotenv
# MongoDB
MONGO_URI=mongodb://user:pass@host:27017/db?authSource=admin
DB_NAME=ollama_chat
COLL_PDF=PDF_
COLL_BIN=Arq_PDF
GRIDFS_BUCKET=fs

# PostgreSQL
PG_HOST=192.168.3.32
PG_PORT=5432
PG_DB=vector_store
PG_USER=vector_store
PG_PASSWORD=senha
PG_SCHEMA=public

# Modelos e dimensões
OLLAMA_EMBEDDING_MODEL=mxbai-embed-large
DIM_MXBAI=1024
SERAFIM_EMBEDDING_MODEL=PORTULAN/serafim-900m-portuguese-pt-sentence-encoder
DIM_SERAFIM=1536
MPNET_EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
DIM_MPNET=768
MINILM_L6_V2=sentence-transformers/all-MiniLM-L6-v2
DIM_MINILM_L6=384
MINILM_L12_V2=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
DIM_MINIL12=384

# Outros
OCR_THRESHOLD=100
CHUNK_SIZE=1024
CHUNK_OVERLAP=700
SLIDING_WINDOW_OVERLAP_RATIO=0.25
MAX_SEQ_LENGTH=128
```

---

## Exemplo de Commit

```text
feat(cli): adiciona modelo MPNet-EN e schema vector_768

- .env: inclui MPNET_EMBEDDING_MODEL e DIM_MPNET
- config.py: carrega novos modelos e dimensões
- main.py: atualiza menu para opção MPNet e schema vector_768
- SQL: adiciona tabela/vector dimension=”768”
```
