# serve.py (Embedding HTTP Server)
#!/usr/bin/env python3
"""
serve.py – Embedding HTTP Server

Exponha embeddings dos modelos Sentence-Transformers via FastAPI.
Por padrão, utiliza o modelo Serafim configurado em SERAFIM_EMBEDDING_MODEL.

Uso:
  pip install fastapi uvicorn sentence-transformers torch
  uvicorn serve:app --host 0.0.0.0 --port 11435
  ou python3 serve.py
"""
import os
from typing import Union, List, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
import uvicorn

# Porta padrão
DEFAULT_PORT = int(os.getenv("EMBEDDING_SERVER_PORT", "11435"))

# Modelo Serafim (corrigido)
SERAFIM_EMBEDDING_MODEL = os.getenv(
    "SERAFIM_EMBEDDING_MODEL",
    "PORTULAN/serafim-900m-portuguese-pt-sentence-encoder"
)

app = FastAPI(title="Embedding Server")

# Cache de instâncias
_loaded_models: Dict[str, SentenceTransformer] = {}

class EmbeddingRequest(BaseModel):
    model: str = Field(
        default=SERAFIM_EMBEDDING_MODEL,
        description="Nome do modelo SentenceTransformer (padrão: Serafim)"
    )
    input: Union[str, List[str]] = Field(
        ..., description="Texto único ou lista de textos para embedar"
    )

class EmbeddingResponse(BaseModel):
    embedding: Union[List[float], List[List[float]]]

def get_model(model_name: str) -> SentenceTransformer:
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
    model = get_model(request.model)
    try:
        embeddings = model.encode(request.input, convert_to_numpy=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao gerar embedding: {e}")
    try:
        emb_list = embeddings.tolist()
    except:
        emb_list = list(embeddings)
    return EmbeddingResponse(embedding=emb_list)

if __name__ == "__main__":
    uvicorn.run("serve:app", host="0.0.0.0", port=DEFAULT_PORT, log_level="info")