#!/usr/bin/env python3
"""
serve.py – Embedding HTTP Server

FastAPI service que converte texto em embeddings.
Em __main__, exibe um menu CLI para selecionar o modelo padrão
e a porta antes de subir o Uvicorn. Carrega sempre em CPU por padrão para
evitar OOM em GPU.
"""

import os
import sys
import logging
import uvicorn
import torch
from typing import List, Union, Optional, Dict

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# ─── Carrega variáveis de ambiente do arquivo .env ──────────────────────────
load_dotenv()

EMBEDDING_MODELS = [
    m.strip()
    for m in os.getenv("EMBEDDING_MODELS", "").split(",")
    if m.strip()
]

DEFAULT_MODEL = os.getenv(
    "DEFAULT_EMBEDDING_MODEL",
    EMBEDDING_MODELS[0] if EMBEDDING_MODELS else None
)

SERVER_PORT = int(os.getenv("EMBEDDING_SERVER_PORT", "11435"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# ─── Configura logging ───────────────────────────────────────────────────────
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)

# ─── Cria a aplicação FastAPI e cache de modelos ────────────────────────────
app = FastAPI(title="Embedding Server")
_model_cache: Dict[str, SentenceTransformer] = {}

def get_model(name: str) -> SentenceTransformer:
    """
    Carrega e cacheia SentenceTransformer em CPU (device='cpu').
    Se falhar, retorna HTTPException(400) indicando modelo inválido.
    """
    if name not in _model_cache:
        try:
            logger.info(f"Carregando modelo '{name}' em CPU...")
            # FORÇAR carregamento em CPU:
            _model_cache[name] = SentenceTransformer(name, device="cpu")
            logger.info(f"Modelo '{name}' carregado com sucesso (device=cpu).")
        except Exception as e:
            logger.error(f"Falha ao carregar modelo '{name}': {e}")
            raise HTTPException(status_code=400, detail=f"Modelo inválido: {name}")
    return _model_cache[name]

# ─── Schemas Pydantic ────────────────────────────────────────────────────────
class EmbeddingRequest(BaseModel):
    model: Optional[str] = Field(
        None, description="Nome do modelo (opcional). Se omitido, usa o padrão."
    )
    input: Union[str, List[str]] = Field(
        ..., description="Texto ou lista de textos para converter."
    )

class EmbeddingResponse(BaseModel):
    embedding: Union[List[float], List[List[float]]]

# ─── Endpoints ──────────────────────────────────────────────────────────────
@app.get("/api/models", response_model=List[str])
async def list_models():
    """Retorna a lista de modelos disponíveis."""
    return EMBEDDING_MODELS

@app.post("/api/embeddings", response_model=EmbeddingResponse)
async def embed(req: EmbeddingRequest, request: Request):
    """
    Gera embeddings para o texto fornecido.
    Usa req.model ou DEFAULT_MODEL.
    Carregamento forçado em CPU (device='cpu').
    """
    model_name = req.model or DEFAULT_MODEL
    if model_name not in EMBEDDING_MODELS:
        raise HTTPException(
            status_code=400, detail=f"Modelo '{model_name}' não disponível."
        )

    # Obtém (ou carrega) o modelo em CPU:
    model = get_model(model_name)

    try:
        # Realmente gera o vetor (em CPU)
        vec = model.encode(req.input, convert_to_numpy=True)
    except Exception as e:
        # Qualquer exceção durante encode gera 500
        logger.error(f"Erro ao gerar embeddings com o modelo '{model_name}': {e}")
        raise HTTPException(status_code=500, detail="Falha ao gerar embeddings.")

    # Converte numpy array em lista de floats
    data = vec.tolist() if hasattr(vec, "tolist") else list(vec)
    return EmbeddingResponse(embedding=data)

@app.get("/health")
async def health(request: Request):
    """Health check básico."""
    return {
        "status": "ok",
        "default_model": DEFAULT_MODEL
    }

# ─── Função de menu para __main__ ────────────────────────────────────────────
def choose_default_model() -> str:
    """
    Exibe um menu de seleção de modelo e retorna o escolhido.
    Só roda em __main__, evitando duplicação de lógica.
    """
    print("\n=== Selecione o modelo padrão de embedding ===")
    for idx, name in enumerate(EMBEDDING_MODELS, start=1):
        tag = " (padrão)" if name == DEFAULT_MODEL else ""
        print(f" {idx}. {name}{tag}")

    choice = input(
        f"Escolha [1-{len(EMBEDDING_MODELS)}] ou ENTER para manter '{DEFAULT_MODEL}': "
    ).strip()

    if choice.isdigit():
        i = int(choice) - 1
        if 0 <= i < len(EMBEDDING_MODELS):
            return EMBEDDING_MODELS[i]
    return DEFAULT_MODEL

# ─── Execução como script ───────────────────────────────────────────────────
if __name__ == "__main__":
    # Antes de subir o servidor, exibe menu para selecionar DEFAULT_MODEL
    if EMBEDDING_MODELS:
        DEFAULT_MODEL = choose_default_model()
        print(f"\nModelo padrão definido: {DEFAULT_MODEL}\n")

    # Solicita porta ao usuário (usa SERVER_PORT como valor padrão)
    try:
        port_input = input(f"Digite a porta para o servidor [{SERVER_PORT}]: ").strip()
        if port_input.isdigit():
            SERVER_PORT = int(port_input)
    except Exception:
        pass

    uvicorn.run(
        "serve:app",
        host="0.0.0.0",
        port=SERVER_PORT,
        log_level=LOG_LEVEL.lower()
    )
