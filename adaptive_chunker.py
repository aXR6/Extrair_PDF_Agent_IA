import os
import logging
import re
from typing import List, Optional
import torch
import nltk
from nltk.corpus import wordnet

from config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    SEPARATORS,
    SLIDING_WINDOW_OVERLAP_RATIO,
    SBERT_MODEL_NAME,
    SERAFIM_EMBEDDING_MODEL,
    MAX_SEQ_LENGTH
)
from utils import filter_paragraphs
from langchain.text_splitter import TokenTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from transformers.utils import logging as tf_logging

# Suprime avisos transformers
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
tf_logging.set_verbosity_error()

# Garante que o WordNet esteja disponível
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

# Cache de instâncias SBERT e Cross-Encoder
_SBERT_CACHE: dict = {}
_CROSS_ENCODER_CACHE: dict = {}

def get_sbert_model(model_name: str) -> SentenceTransformer:
    """
    Retorna instância de SentenceTransformer em cache ou carrega e cacheia.
    Em caso de falha de OSError, exibe mensagem clara e interrompe.
    """
    if model_name not in _SBERT_CACHE:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            _SBERT_CACHE[model_name] = SentenceTransformer(model_name, device=device)
        except OSError as e:
            logging.error(
                f"Não foi possível carregar SBERT '{model_name}': {e}.\n"
                f"Verifique se o identificador está correto e se há conexão ou cache local."
            )
            raise
    return _SBERT_CACHE[model_name]

def get_cross_encoder(model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
    """
    Retorna um CrossEncoder para re-ranking de pares (query, documento).
    """
    if model_name not in _CROSS_ENCODER_CACHE:
        from sentence_transformers import CrossEncoder
        _CROSS_ENCODER_CACHE[model_name] = CrossEncoder(model_name)
    return _CROSS_ENCODER_CACHE[model_name]

# Pré-carrega modelo SBERT default (MiniLM) na importação
try:
    _ = get_sbert_model(SBERT_MODEL_NAME)
except Exception:
    raise

# Funções auxiliares (transform_content, sliding_window_chunk, expand_query)
def transform_content(section: str) -> str:
    words = section.split()
    if len(words) <= 10:
        return section
    input_len = len(words)
    max_len = min(150, max(10, input_len // 2))
    min_len = max(5, int(max_len * 0.25))

    try:
        summary = get_summarizer()(section, max_length=max_len, min_length=min_len, truncation=True)[0]['summary_text']
    except Exception as e:
        logging.warning(f"Sumarização falhou: {e}")
        summary = section
    try:
        ents = get_ner()(section)
        ent_str = '; '.join({e['word'] for e in ents})
    except Exception as e:
        logging.warning(f"NER falhou: {e}")
        ent_str = ''
    try:
        para = get_paraphraser()(summary, max_length=max_len)[0]['generated_text']
    except Exception as e:
        logging.warning(f"Paráfrase falhou: {e}")
        para = summary

    header = f"Entities: {ent_str}\n" if ent_str else ''
    enriched = f"{header}Paraphrase: {para}\nOriginal: {section}"
    return enriched

def sliding_window_chunk(p: str, window_size: int, overlap: int) -> List[str]:
    tokens = p.split()
    stride = max(1, window_size - overlap)
    chunks = []
    for i in range(0, len(tokens), stride):
        part = tokens[i:i + window_size]
        if not part:
            break
        chunks.append(' '.join(part))
        if i + window_size >= len(tokens):
            break
    return chunks

def expand_query(text: str, top_k: int = 5) -> str:
    terms = []
    try:
        for token in set(text.lower().split()):
            syns = wordnet.synsets(token)
            if syns:
                lemmas = {l.name().replace('_', ' ') for s in syns for l in s.lemmas()}
                terms.extend(list(lemmas)[:top_k])
    except Exception as e:
        logging.warning(f"Expansão de query falhou: {e}")
    return text + ' ' + ' '.join(terms)

# Função principal de chunking semântico, dinâmica por modelo
def hierarchical_chunk(
    text: str,
    metadata: dict,
    chunk_model_name: Optional[str] = None
) -> List[str]:
    # escolha dinâmica do modelo para chunking
    use_model = chunk_model_name if chunk_model_name == SERAFIM_EMBEDDING_MODEL else SBERT_MODEL_NAME
    sbert = get_sbert_model(use_model)
    try:
        max_tokens = getattr(sbert, "max_seq_length", sbert.tokenizer.model_max_length)
    except Exception:
        max_tokens = MAX_SEQ_LENGTH

    query = metadata.get('__query')
    if query:
        metadata['__query_expanded'] = expand_query(query)

    final_chunks: List[str] = []
    paras = filter_paragraphs(text)
    clean_text = "\n\n".join(paras)
    sections = semantic_fine_sections(clean_text)

    for sec in sections:
        enriched = transform_content(sec)
        tokens = sbert.tokenizer.tokenize(enriched)
        if len(tokens) <= max_tokens:
            final_chunks.append(enriched)
        else:
            overlap = int(max_tokens * SLIDING_WINDOW_OVERLAP_RATIO)
            parts = sliding_window_chunk(enriched, max_tokens, overlap)
            if not parts:
                parts = TokenTextSplitter(
                    separators=SEPARATORS,
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=overlap
                ).split_text(enriched)
            final_chunks.extend(parts)

    return final_chunks
