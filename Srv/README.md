# Embedding Server

Este repositório contém um microserviço HTTP desenvolvido com [FastAPI](https://fastapi.tiangolo.com/) para gerar embeddings de texto usando modelos da biblioteca [Sentence Transformers](https://www.sbert.net/). O serviço expõe endpoints REST para listar modelos, gerar embeddings e health check, além de um menu CLI para seleção interativa do modelo padrão.

---

## Índice

- [Embedding Server](#embedding-server)
  - [Índice](#índice)
  - [Funcionalidades](#funcionalidades)
  - [Requisitos](#requisitos)
    - [Principais dependências (ver `requirements.txt`):](#principais-dependências-ver-requirementstxt)
  - [Instalação](#instalação)
  - [Variáveis de Ambiente](#variáveis-de-ambiente)
  - [Estrutura do Projeto](#estrutura-do-projeto)
  - [Como Executar](#como-executar)
    - [Selecionando o Modelo Padrão](#selecionando-o-modelo-padrão)
    - [Subindo o Servidor](#subindo-o-servidor)
  - [Endpoints da API](#endpoints-da-api)
    - [Listar Modelos Disponíveis](#listar-modelos-disponíveis)
    - [Gerar Embeddings](#gerar-embeddings)
    - [Health Check](#health-check)
  - [Exemplos de Uso](#exemplos-de-uso)
    - [1. Listar Modelos](#1-listar-modelos)
    - [2. Gerar Embedding (texto único)](#2-gerar-embedding-texto-único)
    - [3. Gerar Embeddings (vários textos)](#3-gerar-embeddings-vários-textos)
    - [4. Health Check](#4-health-check)
  - [Personalização / Ajustes](#personalização--ajustes)
  - [Licença](#licença)

---

## Funcionalidades

- **Listagem de modelos:** Endpoint para retornar a lista de modelos de embedding configurados via variável de ambiente.
- **Geração de embeddings:** Gera embeddings a partir de texto(s) usando Sentence Transformers, carregando os modelos no dispositivo escolhido (`cpu`, `gpu` ou `auto`).
- **Health check:** Retorna status do serviço e o modelo padrão configurado.
- **Seleção interativa de modelo:** Menu CLI para escolher o modelo padrão antes de iniciar o servidor.
- **Seleção de dispositivo:** No processamento local (`main.py`) é possível definir `cpu`, `gpu` ou `auto`.
- **Métricas Prometheus:** chame `start_metrics_server()` para disponibilizar `/metrics` na porta 8000.

---

## Requisitos

- Python 3.8+
- GPU (opcional, recomendado para alto desempenho)
- Acesso à internet (para download dos modelos na primeira execução)

### Principais dependências (ver `requirements.txt`):

- fastapi
- uvicorn
- torch
- sentence-transformers
- python-dotenv
- pydantic

---

## Instalação

Clone o repositório:

```bash
git clone https://github.com/aXR6/Extrair_PDF_Agent_IA/Srv
cd Extrair_PDF_Agent_IA/Srv
```

(Opcional) Crie e ative um ambiente virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Instale as dependências:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Crie um arquivo `.env` na raiz do projeto com as variáveis de configuração (veja exemplo abaixo).

---

## Variáveis de Ambiente

Exemplo de `.env`:

```env
# Lista de modelos de embedding separados por vírgula
EMBEDDING_MODELS=mixedbread-ai/mxbai-embed-large-v1,PORTULAN/serafim-900m-portuguese-pt-sentence-encoder-ir,sentence-transformers/all-MiniLM-L6-v2,sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2,sentence-transformers/all-mpnet-base-v2

# Modelo padrão
DEFAULT_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Porta do servidor
EMBEDDING_SERVER_PORT=11435

# Nível de log
LOG_LEVEL=INFO
```

- `EMBEDDING_MODELS`: identificadores dos modelos aceitos (nomes válidos do SentenceTransformers/Hugging Face).
- `DEFAULT_EMBEDDING_MODEL`: modelo padrão se não especificado no payload.
- `EMBEDDING_SERVER_PORT`: porta TCP do servidor.
- `LOG_LEVEL`: nível de logs (`DEBUG`, `INFO`, `WARNING`, `ERROR`).

> **Nota:** Não versionar o `.env` se contiver informações sensíveis.

---

## Estrutura do Projeto

```
embedding-server/
├── serve.py
├── requirements.txt
├── .env.example
├── README.md
└── .gitignore
```

- **serve.py:** Implementação do servidor FastAPI e menu CLI.
- **requirements.txt:** Dependências Python.
- **.env.example:** Exemplo de configuração.
- **README.md:** Documentação.
- **.gitignore:** Ignora arquivos indesejados.

---

## Como Executar

### Selecionando o Modelo Padrão

Ao rodar o script, será exibido um menu para escolher o modelo padrão:

```bash
python3 serve.py
```

Exemplo de menu:

```
=== Selecione o modelo padrão de embedding ===
 1. mixedbread-ai/mxbai-embed-large-v1
 2. PORTULAN/serafim-900m-portuguese-pt-sentence-encoder-ir (padrão)
 3. sentence-transformers/all-MiniLM-L6-v2
 4. sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
 5. sentence-transformers/all-mpnet-base-v2
Escolha [1-5] ou ENTER para manter 'PORTULAN/serafim-900m-portuguese-pt-sentence-encoder-ir':
```

Após a seleção, o servidor será iniciado.

### Subindo o Servidor

Para iniciar o servidor FastAPI (Uvicorn):

```bash
python3 serve.py
```

Ou, para rodar diretamente (sem menu):

```bash
uvicorn serve:app --host 0.0.0.0 --port 11435 --log-level info
```

---

## Endpoints da API

Todos os endpoints recebem e retornam JSON.

### Listar Modelos Disponíveis

```
GET /api/models
```

**Resposta:**

```json
[
    "mixedbread-ai/mxbai-embed-large-v1",
    "PORTULAN/serafim-900m-portuguese-pt-sentence-encoder-ir",
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "sentence-transformers/all-mpnet-base-v2"
]
```

---

### Gerar Embeddings

```
POST /api/embeddings
```

**Headers:**

```
Content-Type: application/json
```

**Body (exemplo):**

```json
{
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "input": "Um texto de exemplo para gerar embedding"
}
```

Ou para múltiplos textos:

```json
{
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "input": [
        "Primeiro texto",
        "Segundo texto",
        "Outro exemplo"
    ]
}
```

**Resposta:**

Para texto único:

```json
{
    "embedding": [0.123456789, -0.234567891, 0.345678912, ...]
}
```

Para múltiplos textos:

```json
{
    "embedding": [
        [0.12, -0.23, 0.34, ...],
        [0.56, -0.45, 0.78, ...],
        ...
    ]
}
```

**Erros Possíveis:**

- `400 Bad Request`: modelo inválido ou payload incorreto.
- `500 Internal Server Error`: erro ao gerar embedding.

---

### Health Check

```
GET /health
```

**Resposta:**

```json
{
    "status": "ok",
    "default_model": "sentence-transformers/all-MiniLM-L6-v2"
}
```

---

## Exemplos de Uso

### 1. Listar Modelos

```bash
curl -X GET http://localhost:11435/api/models
```

### 2. Gerar Embedding (texto único)

```bash
curl -X POST http://localhost:11435/api/embeddings \
    -H "Content-Type: application/json" \
    -d '{"model": "sentence-transformers/all-MiniLM-L6-v2", "input": "Olá, este é um exemplo de texto para gerar embedding."}'
```

### 3. Gerar Embeddings (vários textos)

```bash
curl -X POST http://localhost:11435/api/embeddings \
    -H "Content-Type: application/json" \
    -d '{"model": "sentence-transformers/all-MiniLM-L6-v2", "input": ["Primeiro exemplo de texto.", "Segundo exemplo, um pouco mais longo.", "Terceiro texto para embeddings."]}'
```

### 4. Health Check

```bash
curl -X GET http://localhost:11435/health
```

---

## Personalização / Ajustes

- **Forçar GPU:** Ajuste a função de carregamento do modelo para usar `device="cuda:0"` se desejar rodar na GPU (atenção à memória disponível).
- **Batch size:** Para listas grandes de textos, ajuste o tamanho do lote (batch size) conforme necessário.
- **Autenticação:** Para produção, implemente autenticação (API Key, OAuth, JWT) ou utilize um proxy reverso.
- **Dockerização:** Exemplo de `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && \
        pip install --no-cache-dir -r requirements.txt

COPY serve.py .
COPY .env .env

EXPOSE 11435

CMD ["python3", "serve.py"]
```

Build e execução:

```bash
docker build -t embedding-server:latest .
docker run -d -p 11435:11435 --name embed-server embedding-server:latest
```

---

## Licença

Consulte o arquivo [LICENSE](LICENSE) para mais informações.

