# 🧠 Extração, Chunking e Indexação Inteligente de Documentos PDF/DOCX

## Visão Geral

Este projeto implementa um pipeline completo para:

- **Extração de texto**: múltiplas estratégias (PyPDF, PDFMiner, OCR, Tika, Unstructured, PyMuPDF4LLM).
- **Chunking inteligente**: técnicas hierárquicas, sliding-window, sumarização, NER, paráfrase e expansão de queries.
- **Indexação e buscas**: embeddings vetoriais (Ollama ou SBERT), busca híbrida (RAG) em PostgreSQL/pgvector e MongoDB.
- **Re-ranking**: Cross-Encoder para aumentar precisão dos resultados.
- **Monitoramento**: métricas em Prometheus (queries, latência, resultados).
- **CLI interativo**: menu para seleção de estratégia, SGBD, schema, modelo e batch processing de pastas.

O sistema processa recursivamente todas as subpastas e pastas "irmãs" do diretório raiz informado, garantindo cobertura completa de documentos.

---

## Funcionalidades

### Extração de Texto

- Detecção automática de PDFs criptografados e fallback OCR.
- Suporte a:
    - PyPDFLoader
    - PDFMinerLoader
    - PDFMiner Low-Level (pdfminer.six)
    - Unstructured (.docx)
    - OCR (pytesseract)
    - PDFPlumber
    - Apache Tika
    - PyMuPDF4LLM (Markdown)

### Chunking Inteligente

- **Filtro de parágrafos**: remove sumários, índices e trechos curtos.
- **Divisão Hierárquica**: detecta títulos seccionados (`\d+(?:\.\d+)*`) para chunks semânticos.
- **Enriquecimento**:
    - Sumarização (DistilBART)
    - Reconhecimento de Entidades (NER)
    - Paráfrase (T5)
- **Sliding Window**: garante sobreposição controlada entre chunks grandes.
- **TokenTextSplitter**: usa separadores customizáveis para dividir caso exceda limites.
- **Padding Tokens**: embeddings são truncados ou preenchidos com zeros para manter dimensão estável.
- **Expansão de Query**: sinônimos via WordNet e injeção em metadata `__query_expanded`.

### Indexação e Buscas

- **MongoDB**:
    - Metadados, textos chunked e arquivos binários em GridFS.
- **PostgreSQL**:
    - Tabela `public.documents` com colunas: `content`, `metadata` (JSONB), `embedding` (VECTOR), `tsv_full`.
    - Triggers: `tsvector_update_trigger` mantém `tsv_full` atualizado.
    - Índices:
        - Vetoriais (HNSW/IVFFlat) via pgvector
        - Full-text GIN em `tsv_full`
        - Trigramas (pg_trgm) em metadata
- **Funções de Busca**:
    - `match_documents_hybrid()` para busca híbrida RAG
    - `match_documents_precise()` para buscas precisas

### Re-ranking e Métricas

- Cross-Encoder para re-ranking pós-embeddings.
- Prometheus Client:
    - `QUERY_EXECUTIONS`, `QUERY_DURATION`, `LAST_QUERY_RESULT_COUNT`.
    - Servidor HTTP `/metrics` (porta 8000).

### CLI e Batch Processing

- Menu interativo para configurar:
    - Estratégia de extração
    - Banco de dados (MongoDB/PostgreSQL)
    - Schema PostgreSQL
    - Modelo de embedding e dimensão
- Processar arquivo ou pasta:
    - Busca recursiva em subpastas e pastas irmãs.
    - Barra de progresso (tqdm).

---

## Técnica de Chunking e Padding

- Os chunks são gerados com tamanho máximo definido por `CHUNK_SIZE` e overlap `CHUNK_OVERLAP`.
- Sliding Window aplica overlap percentual (`SLIDING_WINDOW_OVERLAP_RATIO`) ao dividir tokens quando necessário.
- Ao gerar embeddings, usamos `SentenceTransformer.encode` e, se o vetor tiver tamanho diferente de `dim`, ele é:
    - Truncado se maior que `dim`.
    - Preenchido com zeros para operar no mesmo espaço dimensional, assegurando consistência nos índices pgvector.

---

## Instalação

1. Clone o repositório:
        ```sh
        git clone https://github.com/seu_usuario/seu_projeto.git
        cd seu_projeto
        ```
2. Configure variáveis no `.env`.
3. Instale dependências:
        ```sh
        pip install -r requirements.txt
        ```
4. Configure corpora NLTK (via pacote OS ou download manual).
5. Aplique DDLs em PostgreSQL para habilitar `pgvector`, trigramas e criar funções.
6. Execute métricas (iniciado automaticamente no import de `metrics.py`).
7. Rode o CLI:
        ```sh
        python3 main.py
        ```

---

## Exemplo de Commit

```text
feat(cli): varredura recursiva de raiz e pastas irmãs

- main.py: ao processar pasta, agora inclui:
    • Pasta raiz informada
    • Todas as pastas irmãs no mesmo nível
    • Busca recursiva em cada diretório usando os.walk
- README.md: documentada nova funcionalidade de processamento em lote recursivo
```
