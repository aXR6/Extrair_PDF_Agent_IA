import logging
import os
import re
import tempfile
from typing import List

import fitz
import pikepdf
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

def filter_paragraphs(text: str) -> List[str]:
    """
    Descarta:
      - Sumário / índice (palavras-chave)
      - Linhas estilo ToC: '1.2 Título .................. 17'
      - Trechos muito curtos (<50 chars)
    """
    paras = [p.strip() for p in text.split('\n\n') if p.strip()]
    result = []
    toc_pattern = re.compile(r'^\d+(?:\.\d+)*\s+.+\s+\d+$')
    for p in paras:
        low = p.lower()
        if re.search(r'\b(sum[aá]rio|índice|table of contents|contents?)\b', low):
            continue
        if toc_pattern.match(p):
            continue
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

def repair_pdf(path: str) -> str:
    """
    Tenta consertar o PDF em duas etapas:
      1) pikepdf/QPDF
      2) Ghostscript
    Retorna o caminho para um arquivo temporário reparado, ou
    o path original em caso de falha.
    """
    # etapa 1: pikepdf
    try:
        tmp1 = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        with pikepdf.Pdf.open(path) as pdf:
            pdf.save(tmp1.name)
        return tmp1.name
    except Exception as e:
        logging.warning(f"pikepdf falhou em '{path}': {e}")

    # etapa 2: Ghostscript
    try:
        tmp2 = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        cmd = [
            "gs",
            "-q",                   # silencioso
            "-dNOPAUSE",
            "-dBATCH",
            "-sDEVICE=pdfwrite",
            "-dCompatibilityLevel=1.4",
            "-dPDFSETTINGS=/prepress",
            f"-sOutputFile={tmp2.name}",
            path
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return tmp2.name
    except Exception as e:
        logging.warning(f"Ghostscript falhou em '{path}': {e}")

    # se tudo falhar, retorna original
    return path