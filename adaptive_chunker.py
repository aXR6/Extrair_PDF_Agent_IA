# adaptive_chunker.py
import os
import logging
import re
from typing import List
import torch
import nltk
from nltk.corpus import wordnet

from config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    SEPARATORS,
    SLIDING_WINDOW_OVERLAP_RATIO,
    SBERT_MODEL_NAME,
    MAX_SEQ_LENGTH
)
from utils import filter_paragraphs
from langchain.text_splitter import TokenTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from transformers.utils import logging as tf_logging

# suppress transformers advisory warnings
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
tf_logging.set_verbosity_error()

# ensure WordNet is available
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

# caches
_SBERT_CACHE: dict = {}
_CROSS_ENCODER_CACHE: dict = {}

def get_sbert_model(model_name: str) -> SentenceTransformer:
    """
    Load (or retrieve from cache) a SentenceTransformer by name.
    """
    if model_name not in _SBERT_CACHE:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            _SBERT_CACHE[model_name] = SentenceTransformer(model_name, device=device)
        except OSError as e:
            logging.error(
                f"Could not load SBERT '{model_name}': {e}. "
                "Check that identifier or local cache."
            )
            raise
    return _SBERT_CACHE[model_name]

def get_cross_encoder(model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
    """
    Load (or retrieve from cache) a CrossEncoder for reranking.
    """
    if model_name not in _CROSS_ENCODER_CACHE:
        from sentence_transformers import CrossEncoder
        _CROSS_ENCODER_CACHE[model_name] = CrossEncoder(model_name)
    return _CROSS_ENCODER_CACHE[model_name]

def semantic_fine_sections(text: str) -> List[str]:
    """
    Split on numeric headings (e.g. "1.2 Title") to preserve semantic sections.
    """
    pattern = re.compile(r'^(?P<heading>\d+(?:\.\d+)*\s+.+)$', re.MULTILINE)
    splits = pattern.split(text)
    if len(splits) < 3:
        return [text]
    sections: List[str] = []
    for i in range(1, len(splits), 2):
        heading = splits[i].strip()
        content = splits[i+1].strip() if i+1 < len(splits) else ''
        sections.append(f"{heading}\n{content}")
    return sections

# we keep no global SBERT instance — chunking may pick any model dynamically
# define summarizer/NER/paraphraser pipelines lazily
_summarizer = None
_ner = None
_paraphraser = None

def get_summarizer():
    global _summarizer
    if _summarizer is None:
        _summarizer = pipeline(
            'summarization',
            model='sshleifer/distilbart-cnn-12-6',
            revision='a4f8f3e',
            device=-1
        )
    return _summarizer

def get_ner():
    global _ner
    if _ner is None:
        _ner = pipeline(
            'ner',
            model='dbmdz/bert-large-cased-finetuned-conll03-english',
            aggregation_strategy='simple',
            device=-1
        )
    return _ner

def get_paraphraser():
    global _paraphraser
    if _paraphraser is None:
        _paraphraser = pipeline(
            'text2text-generation',
            model='t5-small',
            device=-1
        )
    return _paraphraser

def transform_content(section: str) -> str:
    words = section.split()
    if len(words) <= 10:
        return section
    input_len = len(words)
    max_len = min(150, max(10, input_len // 2))
    min_len = max(5, int(max_len * 0.25))
    try:
        summary = get_summarizer()(section, max_length=max_len, min_length=min_len, truncation=True)[0]['summary_text']
    except Exception:
        summary = section
    try:
        ents = get_ner()(section)
        ent_str = '; '.join({e['word'] for e in ents})
    except Exception:
        ent_str = ''
    try:
        para = get_paraphraser()(summary, max_length=max_len)[0]['generated_text']
    except Exception:
        para = summary
    header = f"Entities: {ent_str}\n" if ent_str else ''
    return f"{header}Paraphrase: {para}\nOriginal: {section}"

def sliding_window_chunk(p: str, window_size: int, overlap: int) -> List[str]:
    tokens = p.split()
    stride = max(1, window_size - overlap)
    chunks: List[str] = []
    for i in range(0, len(tokens), stride):
        part = tokens[i : i + window_size]
        if not part:
            break
        chunks.append(' '.join(part))
        if i + window_size >= len(tokens):
            break
    return chunks

def expand_query(text: str, top_k: int = 5) -> str:
    terms: List[str] = []
    try:
        for token in set(text.lower().split()):
            syns = wordnet.synsets(token)
            if syns:
                lemmas = {l.name().replace('_', ' ') for s in syns for l in s.lemmas()}
                terms.extend(list(lemmas)[:top_k])
    except Exception:
        pass
    return text + ' ' + ' '.join(terms)

def hierarchical_chunk(
    text: str,
    metadata: dict,
    chunk_model_name: str = None
) -> List[str]:
    """
    Create semantic chunks using:
      • filter_paragraphs
      • semantic_fine_sections
      • optional summarization/NER/paraphrase
      • token cutoff + sliding-window

    If chunk_model_name provided, SBERT for token count will be loaded from that,
    otherwise falls back to SBERT_MODEL_NAME from config.
    """
    model_name = chunk_model_name or SBERT_MODEL_NAME
    sb_model = get_sbert_model(model_name)

    # expand query if any
    query = metadata.get('__query')
    if query:
        metadata['__query_expanded'] = expand_query(query)

    fragments: List[str] = []
    paras = filter_paragraphs(text)
    joined = "\n\n".join(paras)
    sections = semantic_fine_sections(joined)

    for sec in sections:
        enriched = transform_content(sec)
        tokens = sb_model.tokenizer.tokenize(enriched)
        if len(tokens) <= getattr(sb_model, "max_seq_length", MAX_SEQ_LENGTH):
            fragments.append(enriched)
        else:
            max_ov = int(getattr(sb_model, "max_seq_length", MAX_SEQ_LENGTH) * SLIDING_WINDOW_OVERLAP_RATIO)
            parts = sliding_window_chunk(enriched, getattr(sb_model, "max_seq_length", MAX_SEQ_LENGTH), max_ov)
            if not parts:
                parts = TokenTextSplitter(
                    separators=SEPARATORS,
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=max_ov
                ).split_text(enriched)
            fragments.extend(parts)

    return fragments