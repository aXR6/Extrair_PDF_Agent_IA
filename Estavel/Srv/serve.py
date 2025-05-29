#!/usr/bin/env python3
"""
serve.py – Embedding HTTP Server

FastAPI service that converts text to embeddings.
Em __main__, exibe um menu CLI para selecionar o modelo padrão
antes de subir o Uvicorn. Faz fallback automático para CPU em caso de OOM.
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

# ─── Carrega .env ───────────────────────────────────────────────────────────
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

# ─── Inicializa logging ─────────────────────────────────────────────────────
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)

# ─── Cria FastAPI app ────────────────────────────────────────────────────────
app = FastAPI(title="Embedding Server")
_model_cache: Dict[str, SentenceTransformer] = {}

def get_model(name: str) -> SentenceTransformer:
    """Carrega e cacheia SentenceTransformer."""
    if name not in _model_cache:
        try:
            _model_cache[name] = SentenceTransformer(name)
        except Exception as e:
            logger.error(f"Falha ao carregar modelo '{name}': {e}")
            raise HTTPException(status_code=400, detail=f"Modelo inválido: {name}")
    return _model_cache[name]

# ─── Pydantic Schemas ────────────────────────────────────────────────────────
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
    Faz fallback para CPU em caso de CUDA OOM.
    """
    model_name = req.model or DEFAULT_MODEL
    if model_name not in EMBEDDING_MODELS:
        raise HTTPException(
            status_code=400, detail=f"Modelo '{model_name}' não disponível."
        )

    model = get_model(model_name)
    try:
        vec = model.encode(req.input, convert_to_numpy=True)
    except RuntimeError as e:
        err = str(e).lower()
        if "out of memory" in err:
            logger.warning(f"CUDA OOM em {model_name}, tentando CPU fallback.")
            torch.cuda.empty_cache()
            model.to("cpu")
            try:
                vec = model.encode(req.input, convert_to_numpy=True)
            except Exception as e2:
                logger.error(f"CPU fallback falhou: {e2}")
                raise HTTPException(
                    status_code=500, detail="Erro ao gerar embeddings em CPU."
                )
        else:
            logger.error(f"Erro inesperado no encode: {e}")
            raise HTTPException(status_code=500, detail="Falha ao gerar embeddings.")
    except Exception as e:
        logger.error(f"Erro no encode: {e}")
        raise HTTPException(status_code=500, detail="Falha ao gerar embeddings.")

    data = vec.tolist() if hasattr(vec, "tolist") else list(vec)
    return EmbeddingResponse(embedding=data)

@app.get("/health")
async def health(request: Request):
    """Health check."""
    return {
        "status": "ok",
        "default_model": DEFAULT_MODEL
    }

# ─── Função de menu para __main__ ────────────────────────────────────────────
def choose_default_model() -> str:
    """
    Exibe um menu de seleção de modelo e retorna o escolhido.
    Rodar apenas em __main__ para evitar duplicação.
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
    # Executa menu ANTES de subir o servidor
    if EMBEDDING_MODELS:
        DEFAULT_MODEL = choose_default_model()
        print(f"\nModelo padrão definido: {DEFAULT_MODEL}\n")
    uvicorn.run(
        "serve:app",
        host="0.0.0.0",
        port=SERVER_PORT,
        log_level=LOG_LEVEL.lower()
    )
