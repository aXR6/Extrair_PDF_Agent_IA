#!/usr/bin/env python3
"""
server.py – Embedding HTTP Server

Exponha embeddings do modelos Sentence-Transformers via FastAPI.
Suporte multilíngue com: jvanhoof/all-MiniLM-L6-multilingual-v2-en-es-pt-pt-br-v2

Uso:
  # Instale dependências:
  pip install fastapi uvicorn sentence-transformers torch

  # Execute via CLI do Uvicorn:
  uvicorn server:app --host 0.0.0.0 --port 11435

  # Ou diretamente:
  python3 server.py
"""
import os
from typing import Union, List, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
import uvicorn

# Porta padrão para o servidor HTTP (pode ser configurada por variável de ambiente)
DEFAULT_PORT = int(os.getenv("EMBEDDING_SERVER_PORT", "11435"))

app = FastAPI(title="Embedding Server")

# Cache simples de modelos carregados: model_name -> SentenceTransformer
_loaded_models: Dict[str, SentenceTransformer] = {}

class EmbeddingRequest(BaseModel):
    model: str = Field(..., description="Nome do modelo SentenceTransformer")
    input: Union[str, List[str]] = Field(
        ..., description="Texto único ou lista de textos para embedar"
    )

class EmbeddingResponse(BaseModel):
    embedding: Union[List[float], List[List[float]]]


def get_model(model_name: str) -> SentenceTransformer:
    """
    Carrega e cacheia uma instância de SentenceTransformer.
    Lança HTTPException(400) se falhar.
    """
    if model_name not in _loaded_models:
        try:
            _loaded_models[model_name] = SentenceTransformer(model_name)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Falha ao carregar o modelo '{model_name}': {e}"
            )
    return _loaded_models[model_name]

@app.post("/api/embeddings", response_model=EmbeddingResponse)
async def embed(request: EmbeddingRequest):
    """
    Gera embedding(s) para o(s) texto(s) de entrada.
    - Se `input` for string, retorna lista de floats.
    - Se for lista de strings, retorna lista de listas.
    """
    model = get_model(request.model)
    try:
        embeddings = model.encode(request.input, convert_to_numpy=True)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao gerar embedding: {e}"
        )

    # Converter numpy.ndarray para listas Python
    try:
        emb_list = embeddings.tolist()
    except Exception:
        emb_list = list(embeddings)

    return EmbeddingResponse(embedding=emb_list)

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=DEFAULT_PORT,
        log_level="info"
    )