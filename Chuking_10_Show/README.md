# 🧠 Extração, Chunking e Indexação Inteligente de Documentos PDF/DOCX

## Visão Geral

Este projeto implementa um pipeline completo para extração, processamento, chunking, enriquecimento semântico e indexação de documentos PDF e DOCX em bancos de dados MongoDB e PostgreSQL. Suporta busca semântica via embeddings e busca textual híbrida (RAG).

O sistema integra OCR, NLP, sumarização, NER e múltiplas estratégias de extração, permitindo consultas avançadas e rápidas sobre grandes volumes de documentos técnicos ou acadêmicos.

---

## Funcionalidades Principais

- **Extração de texto avançada** de PDFs/DOCX usando diferentes estratégias (OCR, Tika, PyPDF, Unstructured, PDFMiner, etc.)
- **Chunking inteligente:** divisão dos documentos em seções/chunks, com regras hierárquicas, sliding window, enriquecimento semântico, sumarização, NER e paráfrase.
- **Geração automática de embeddings** (via API do Ollama ou compatível) para cada chunk, permitindo busca vetorial.
- **Indexação de dados:**
    - **MongoDB:** armazenamento de chunks, metadados e arquivos binários.
    - **PostgreSQL:** armazenamento de chunks, metadados, embeddings (usando pgvector) e campo `tsv_full` para busca textual híbrida (RAG).
- **População automática dos índices** (`tsv_full` via trigger).
- **Estrutura de banco preparada para buscas semânticas híbridas** (função `match_documents_hybrid`).
- **Configuração flexível** via `.env`.
- **Interface CLI** com menu para facilitar escolha de estratégia de extração e banco alvo.
- **Extração e enriquecimento de metadados automáticos** dos arquivos.
- **Processamento em lote** de pastas com barra de progresso.
- **Compatível com automações** e integrações com n8n, pipelines ETL, APIs, etc.

---

## Arquitetura dos Códigos

<details>
<summary><strong>main.py</strong></summary>
Responsável pelo fluxo principal do CLI: seleção de estratégia de extração e banco de dados, processamento individual ou em lote, coordenação das etapas e chamada dos módulos auxiliares.
</details>

<details>
<summary><strong>config.py</strong></summary>
Carrega as configurações do projeto a partir do arquivo `.env` e define variáveis globais (URIs, senhas, parâmetros de chunk, OCR, separadores, etc).
</details>

<details>
<summary><strong>extractors.py</strong></summary>
Módulo com todas as estratégias de extração de texto (PyPDF, PDFMiner, Unstructured, OCR, etc.) e utilitário para metadados.
</details>

<details>
<summary><strong>adaptive_chunker.py</strong></summary>
Responsável por dividir o texto extraído em chunks significativos, aplicando chunking hierárquico, sliding window, sumarização, NER e paráfrase.
</details>

<details>
<summary><strong>pg_storage.py</strong></summary>
Integração e escrita dos dados no PostgreSQL, geração de embeddings, inserção de chunks e metadados, uso de pool de conexões.
</details>

<details>
<summary><strong>storage.py</strong></summary>
Persistência no MongoDB, salvando metadados, arquivos binários e integração com GridFS.
</details>

<details>
<summary><strong>utils.py</strong></summary>
Funções auxiliares para logging, validação de arquivos, geração de relatórios, filtragem de parágrafos e chunking recursivo.
</details>

---

## Estrutura do Banco de Dados (PostgreSQL)

- **Tabela `documents`:** cada chunk é uma linha, com conteúdo, metadados (`JSONB`), embedding (`vector`) e `tsv_full`.
- **Triggers:** mantêm `tsv_full` sincronizado automaticamente.
- **Índices:**
    - Vetoriais (IVFFlat, HNSW via pgvector)
    - Full-text (GIN para `tsv_full`)
- **Função `match_documents_hybrid`:** busca híbrida de chunks por similaridade vetorial e score léxico, com pós-filtragem por contexto pai.

---

## Como Usar

### Pré-requisitos

- Python 3.8+
- PostgreSQL 15+ com extensão `pgvector`
- MongoDB (opcional)
- Dependências Python (ver `requirements.txt`)
- API de embeddings (Ollama ou compatível)

### Configuração

1. Configure o `.env` conforme suas credenciais e parâmetros.
2. Garanta que as estruturas SQL estejam aplicadas ao PostgreSQL.
3. Instale as dependências:

     ```sh
     pip install -r requirements.txt
     ```

### Execução

1. Rode o menu principal:

     ```sh
     python3 main.py
     ```

2. Escolha a estratégia de extração (ex: OCR, PyPDF, Unstructured).
3. Escolha o banco de dados (MongoDB ou PostgreSQL).
4. Informe o arquivo ou pasta para processar.
5. Acompanhe o progresso e verifique logs/resultados no banco escolhido.

---

## FAQ e Observações

- Ajuste `chunk_size`/`chunk_overlap` conforme o contexto dos documentos.
- Integração com n8n e outros orchestrators é direta via SQL ou API REST.
- Performance pode ser tunada ajustando parâmetros dos índices.
- Para grandes volumes, use processamento em batch para evitar throttling da API de embedding.
- Em caso de ambiguidade SQL, sempre qualifique o campo com o alias do select/CTE.

---

## Autores & Colaboradores

Projeto desenvolvido para workflows de I.A., automação e pesquisa documental em larga escala, por [Thalles Canela].

---

