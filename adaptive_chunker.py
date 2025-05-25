import os
import logging
import torch
import nltk
from typing import List
from nltk.corpus import wordnet
from config import (
    CHUNK_SIZE, SLIDING_WINDOW_OVERLAP_RATIO,
    SBERT_MODEL_NAME
)
from utils import filter_paragraphs
from sentence_transformers import SentenceTransformer
from transformers.utils import logging as tf_logging

# Supressão de avisos transformers
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
tf_logging.set_verbosity_error()

# Garante que o WordNet esteja disponível
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

# Cache de instâncias SBERT
_SBERT_CACHE: dict = {}

def get_sbert_model(model_name: str = SBERT_MODEL_NAME) -> SentenceTransformer:
    """Carrega e retorna instância SentenceTransformer em cache."""
    if model_name not in _SBERT_CACHE:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            _SBERT_CACHE[model_name] = SentenceTransformer(model_name, device=device)
        except Exception as e:
            logging.error(f"Falha ao carregar SBERT '{model_name}': {e}")
            raise
    return _SBERT_CACHE[model_name]


def expand_query(text: str, top_k: int = 5) -> str:
    """Gera termos de expansão usando sinônimos do WordNet."""
    terms = []
    for token in set(text.lower().split()):
        syns = wordnet.synsets(token)
        for syn in syns[:top_k]:
            for lemma in syn.lemmas()[:top_k]:
                terms.append(lemma.name().replace('_', ' '))
    return text + ' ' + ' '.join(set(terms))


def hierarchical_chunk(text: str, metadata: dict, model_name: str = SBERT_MODEL_NAME) -> List[str]:
    """
    Chunking inteligente baseado em parágrafos:
    - Detecta parágrafos completos via filter_paragraphs.
    - Agrupa parágrafos inteiros até atingir o limite de tokens.
    - Se um parágrafo exceder o limite, divide-o internamente em sub-chunks.
    """
    # Limpa cache da GPU antes
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = get_sbert_model(model_name)
    tokenizer = model.tokenizer
    try:
        max_tokens = model.max_seq_length
    except AttributeError:
        max_tokens = tokenizer.model_max_length

    # Expansão de query
    query = metadata.get('__query')
    if query:
        metadata['__query_expanded'] = expand_query(query)

    paras = filter_paragraphs(text)
    chunks: List[str] = []
    current_para_group: List[str] = []
    current_tok_count = 0

    for para in paras:
        # Conta tokens do parágrafo completo
        tokens = tokenizer.tokenize(para)
        tok_len = len(tokens)

        # Parágrafo sozinho excede o limite
        if tok_len > max_tokens:
            # Fecha grupo atual
            if current_para_group:
                chunks.append("\n\n".join(current_para_group))
                current_para_group, current_tok_count = [], 0
            # Divide o próprio parágrafo em sub-chunks via TokenTextSplitter
            from langchain.text_splitter import TokenTextSplitter
            splitter = TokenTextSplitter(
                chunk_size=max_tokens,
                chunk_overlap=int(max_tokens * SLIDING_WINDOW_OVERLAP_RATIO)
            )
            sub_chunks = splitter.split_text(para)
            chunks.extend(sub_chunks)
            continue

        # Cabe no chunk atual?
        if current_tok_count + tok_len <= max_tokens:
            current_para_group.append(para)
            current_tok_count += tok_len
        else:
            # Fecha chunk atual
            chunks.append("\n\n".join(current_para_group))
            # Inicia novo grupo com este parágrafo
            current_para_group = [para]
            current_tok_count = tok_len

    # Adiciona resto
    if current_para_group:
        chunks.append("\n\n".join(current_para_group))

    # Limpa cache da GPU após
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return chunks