#adaptive_chunker.py
import os
import logging
import re
from typing import List
from config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    SEPARATORS,
    SLIDING_WINDOW_OVERLAP_RATIO,
    SIMILARITY_THRESHOLD
)
from langchain.text_splitter import TokenTextSplitter
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from transformers.utils import logging as tf_logging

# Suppress transformer warnings
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
tf_logging.set_verbosity_error()

# Initialize SBERT model for semantic chunking once
_sbert = SentenceTransformer('all-MiniLM-L6-v2')

# Lazy-loaded pipelines to avoid OOM on import
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
            revision='4c53496',
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

def semantic_fine_sections(text: str) -> List[str]:
    pattern = re.compile(r'^(?P<heading>\d+(?:\.\d+)*\s+.+)$', re.MULTILINE)
    splits = pattern.split(text)
    if len(splits) < 3:
        return [text]
    sections = []
    for i in range(1, len(splits), 2):
        heading = splits[i].strip()
        content = splits[i+1].strip() if i+1 < len(splits) else ''
        sections.append(f"{heading}\n{content}")
    return sections

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

def hierarchical_chunk(text: str, metadata: dict) -> List[str]:
    final_chunks: List[str] = []
    sections = semantic_fine_sections(text)
    for sec in sections:
        enriched = transform_content(sec)
        if len(enriched) <= CHUNK_SIZE:
            final_chunks.append(enriched)
        else:
            max_ov = min(CHUNK_OVERLAP, int(len(enriched) * SLIDING_WINDOW_OVERLAP_RATIO))
            parts = sliding_window_chunk(enriched, CHUNK_SIZE, max_ov)
            if not parts:
                logging.warning(f"Fallback recursive para seção len={len(enriched)}")
                parts = TokenTextSplitter(
                    separators=SEPARATORS,
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=max_ov
                ).split_text(enriched)
            final_chunks.extend(parts)
    return final_chunks