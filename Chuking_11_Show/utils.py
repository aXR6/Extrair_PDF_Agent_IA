# utils.py
import logging
import os
import re
from typing import List
import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP, SEPARATORS

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def build_record(path: str, text: str) -> dict:
    try:
        doc = fitz.open(path)
        info = doc.metadata.copy() or {}
        info['numpages'] = doc.page_count
        doc.close()
    except Exception as e:
        logging.error(f"Erro metadados: {e}")
        info = {}
    return {'text': text, 'info': info, 'version': '2.16.105'}

def is_valid_file(path: str) -> bool:
    if not os.path.isfile(path) or not path.lower().endswith(('.pdf', '.docx')):
        logging.error(f"Arquivo inválido: {path}")
        return False
    return True

def generate_report(processed: List[str], errors: List[str]):
    logging.info(f"Processados: {len(processed)}")
    logging.warning(f"Erros: {len(errors)}")
    for e in errors:
        logging.warning(e)

def filter_paragraphs(text: str) -> List[str]:
    """
    Descarta:
      - Sumário / índice (palavras-chave)
      - Linhas estilo ToC: '1.2 Título .................. 17'
      - Trechos muito curtos (<50 chars)
    """
    paras = [p.strip() for p in text.split('\n\n') if p.strip()]
    result = []
    toc_pattern = re.compile(r'^\d+(?:\.\d+)*\s+.+\s+\d+$')  # ex: '1.2 Seção ................ 17'
    for p in paras:
        low = p.lower()
        # descarta sumário explícito
        if re.search(r'\b(sum[aá]rio|índice|table of contents|contents?)\b', low):
            continue
        # descarta linhas tipo ToC com número de página
        if toc_pattern.match(p):
            continue
        # descarta textos curtos
        if len(p) < 50:
            continue
        result.append(p)
    return result

def chunk_text(text: str, metadata: dict) -> List[str]:
    paras = filter_paragraphs(text)
    chunks: List[str] = []
    for p in paras:
        if len(p) <= CHUNK_SIZE:
            chunks.append(p)
        else:
            overlap = min(CHUNK_OVERLAP, len(p) // 2)
            splitter = RecursiveCharacterTextSplitter(
                separators=SEPARATORS,
                chunk_size=CHUNK_SIZE,
                chunk_overlap=overlap
            )
            sub = splitter.split_text(p)
            chunks.extend(sub or [p])
    return chunks