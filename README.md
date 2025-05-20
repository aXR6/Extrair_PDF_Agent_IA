# 🧠 Extração, Chunking e Indexação Inteligente de Documentos PDF/DOCX

## Visão Geral

Este projeto oferece um pipeline completo para extração, processamento, chunking, enriquecimento semântico e indexação de documentos PDF e DOCX em bancos de dados MongoDB e PostgreSQL. Permite buscas semânticas com embeddings, busca híbrida (RAG), re-ranking e monitoramento de métricas.

O sistema integra OCR, NLP, sumarização, NER, expansão de queries e múltiplas estratégias de extração, possibilitando consultas avançadas e rápidas sobre grandes volumes de documentos técnicos ou acadêmicos.

---

## Funcionalidades

- **Extração avançada de texto** de PDFs/DOCX usando OCR, Tika, PyPDF, Unstructured, PDFMiner, entre outros.
- **Chunking inteligente:** divisão dos documentos em seções/chunks, com regras hierárquicas, sliding window e enriquecimento semântico (sumarização, NER, paráfrase).
- **Expansão de queries:** aplicação de sinônimos e termos relacionados via WordNet para melhorar o recall.
- **Geração automática de embeddings** (Ollama ou SBERT local) para cada chunk, permitindo busca vetorial.
- **Re-ranking com Cross-Encoder:** pós-processamento dos resultados de busca vetorial para maior precisão.
- **Monitoramento de métricas:** coleta de contagem de queries, latência e número de resultados via Prometheus (`/metrics`).
- **Indexação de dados:**
    - **MongoDB:** armazenamento de chunks, metadados e arquivos binários.
    - **PostgreSQL:** armazenamento de chunks, metadados, embeddings (pgvector), `tsv_full` para busca híbrida e função `match_documents_hybrid`.
- **Triggers e índices automáticos:** mantém `tsv_full` sincronizado e utiliza índices vetoriais (HNSW/IVFFlat) e trigramas para metadata.
- **Interface CLI intuitiva:** menu para seleção de estratégia de extração, banco de dados, chunking e embeddings.
- **Processamento em lote** de pastas com barra de progresso.
- **Configuração flexível** via `.env`.
- **Compatível com ETL, API REST e orquestradores** como n8n.

---

## Arquitetura dos Códigos

<details>
<summary><strong>main.py</strong></summary>
Gerencia o fluxo principal do CLI: seleção de estratégias, banco de dados, processamento e coordenação dos módulos.
</details>

<details>
<summary><strong>config.py</strong></summary>
Carrega configurações do projeto a partir do `.env` e define variáveis globais.
</details>

<details>
<summary><strong>extractors.py</strong></summary>
Implementa estratégias de extração de texto (PyPDF, PDFMiner, Unstructured, OCR, etc.) e extração de metadados.
</details>

<details>
<summary><strong>adaptive_chunker.py</strong></summary>
Divide o texto extraído em chunks com chunking hierárquico, sliding window, sumarização, NER, paráfrase e query expansion.
</details>

<details>
<summary><strong>metrics.py</strong></summary>
Coleta métricas de RAG (contagem de queries, latência, número de resultados) e expõe endpoint Prometheus.
</details>

<details>
<summary><strong>pg_storage.py</strong></summary>
Integração com PostgreSQL: geração e inserção de embeddings, chunking semântico, re-ranking com cross-encoder e registro de métricas.
</details>

<details>
<summary><strong>storage.py</strong></summary>
Persistência no MongoDB, salvando metadados, arquivos binários e integração com GridFS.
</details>

<details>
<summary><strong>utils.py</strong></summary>
Funções auxiliares para logging, validação de arquivos, geração de relatórios e filtragem de parágrafos.
</details>

---

## Estrutura do Banco de Dados (PostgreSQL)

- **Tabela `documents`:** cada chunk é uma linha com `content`, `metadata` (`JSONB`), `embedding` (`VECTOR`) e `tsv_full`.
- **Triggers:** mantêm `tsv_full` atualizado automaticamente.
- **Índices:**
    - **Vetoriais:** HNSW e IVFFlat via pgvector.
    - **Full-text:** GIN em `tsv_full`.
    - **Trigramas:** GiST+pg_trgm em campos críticos de metadata.
- **Funções:** `match_documents_hybrid` e `match_documents_precise` para busca híbrida.

---

## Como Usar

### Pré-requisitos

- Python 3.8+
- PostgreSQL 15+ com extensão `pgvector`
- MongoDB (opcional)
- Dependências Python (`requirements.txt`)
- API de embeddings (Ollama ou SBERT local)

### Configuração

1. Ajuste o `.env` com suas credenciais e parâmetros.
2. Aplique as DDLs no PostgreSQL.
3. Instale dependências:

    ```sh
    pip install -r requirements.txt
    ```

### Configuração do NLTK

#### 1. Instalar corpora via APT

Em Debian 12, existem pacotes que fornecem tanto o código do NLTK quanto muitos dos datasets mais usados.

```bash
sudo apt-get update                                 # Atualiza lista de pacotes  
sudo apt-get install -y python3-nltk nltk-data      # Instala o NLTK e os dados via repositório oficial  
```
O pacote `python3-nltk` traz a biblioteca em si e algumas dependências.  
O pacote `nltk-data` inclui boa parte dos corpora “popular” já prontos, evitando downloads via HTTP direto dos servidores do NLTK.

#### 2. Download manual para um diretório local

Se precisar apenas de alguns corpora específicos, você pode baixá-los manualmente e armazenar numa pasta local:

Crie o diretório de dados, por exemplo:

```bash
sudo mkdir -p /usr/local/share/nltk_data/{corpora,tokenizers,taggers}
sudo chown -R $USER: /usr/local/share/nltk_data
```

Baixe pacotes individuais do GitHub do projeto nltk_data. Exemplo:

```bash
cd /usr/local/share/nltk_data/corpora
wget https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/stopwords.zip
unzip stopwords.zip
rm stopwords.zip
```

Aponte o NLTK para esse diretório (opcional, se não estiver no padrão):

```python
import nltk
nltk.data.path.append('/usr/local/share/nltk_data')
```

### Execução

1. Inicie o serviço de métricas (Prometheus) se desejar:

    ```sh
    # já é iniciado automaticamente pelo metrics.py na porta 8000
    ```

2. Rode o menu principal:

    ```sh
    python3 main.py
    ```

3. Selecione a estratégia de extração, banco de dados e informe o arquivo ou pasta.

4. Acompanhe logs, progresso e consulte métricas em [http://localhost:8000/metrics](http://localhost:8000/metrics).

---

## FAQ & Dicas

- Ajuste `min_cos_sim` e `min_score` em suas queries SQL para controlar precisão.
- Para alta precisão, utilize re-ranking e thresholds mais altos.
- Monitore Precision@k e Recall@k via dashboards Prometheus/Grafana.

---

## Commit Breve

**Autor:** Thalles Canela <<thalles@example.com>>  
**Data:** 2025-05-20 12:34:56 -0300

### feat(rag): expansão de queries, re-ranking com cross-encoder e métricas Prometheus

- **adaptive_chunker.py**
    - Implementada a função `expand_query()` usando WordNet para expansão de queries (pseudo-relevance feedback).
    - Integrada a expansão de queries (`__query_expanded`) ao método `hierarchical_chunk()` para enriquecer as buscas.
    - Mantida a lógica existente de chunking e enriquecimento (sumarização, NER, paráfrase).
    - Adicionado helper `get_cross_encoder()` para carregar CrossEncoder do Sentence-Transformers.

- **pg_storage.py**
    - `generate_embedding()` mantido inalterado.
    - Adicionada função `rerank_with_cross_encoder()` para re-ranking dos resultados RAG via cross-encoder.
    - Função `save_to_postgres()` agora decorada com `@record_metrics` (de metrics.py).
    - `save_to_postgres()`:
        - Insere semantic chunks normalmente.
        - Aplica re-ranking com cross-encoder nos chunks inseridos.
        - Retorna documentos reranqueados.

- **metrics.py** (novo)
    - Integração com Prometheus:
        - Métricas: `QUERY_COUNT`, `QUERY_LATENCY`, `RESULT_COUNT`.
        - Decorator `record_metrics` para medição automática.
        - Servidor HTTP na porta 8000 expondo `/metrics`.

- **README.md**
    - Documentação atualizada para:
        - Expansão de queries.
        - Passo de re-ranking com cross-encoder.
        - Monitoramento de métricas com Prometheus.
        - Instruções de uso do endpoint `/metrics`.

- **requirements.txt**
    - Adicionada dependência `prometheus_client`.

Essas mudanças aprimoram a precisão do RAG com pseudo-relevance feedback, re-ranking detalhado e monitoramento em tempo real das métricas de performance das queries.

## Autoria

Desenvolvido por Thalles Canela para workflows de IA e pesquisa documental em larga escala.
