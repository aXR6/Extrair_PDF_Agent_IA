# utils.py
import logging
import os
import re
import tempfile
import subprocess
from typing import List

import fitz
import pikepdf
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP, SEPARATORS
from constants import VALID_EXTS

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
    """Verifica extensão suportada."""
    if not os.path.isfile(path) or not path.lower().endswith(VALID_EXTS):
        logging.error(f"Arquivo inválido: {path}")
        return False
    return True

def filter_paragraphs(text: str) -> List[str]:
    """
    Descarta sumários, índices e trechos curtos (<50 chars).
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
    Tenta consertar o PDF em múltiplas etapas: mutool, pikepdf, Ghostscript.
    Retorna o caminho para um arquivo temporário reparado ou o original.
    """
    # mutool clean
    try:
        tmp0 = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        tmp0.close()
        subprocess.run(
            ["mutool", "clean", "-d", path, tmp0.name],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return tmp0.name
    except Exception as e:
        logging.warning(f"mutool clean falhou em '{path}': {e}")
        try:
            os.remove(tmp0.name)
        except Exception:
            pass

    # pikepdf
    try:
        tmp1 = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        tmp1.close()
        with pikepdf.Pdf.open(path) as pdf:
            pdf.save(tmp1.name)
        return tmp1.name
    except Exception as e:
        logging.warning(f"pikepdf falhou em '{path}': {e}")
        try:
            os.remove(tmp1.name)
        except Exception:
            pass

    # Ghostscript
    try:
        tmp2 = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        tmp2.close()
        cmd = [
            "gs", "-q", "-dNOPAUSE", "-dBATCH",
            "-sDEVICE=pdfwrite", "-dCompatibilityLevel=1.4",
            "-dPDFSETTINGS=/prepress",
            f"-sOutputFile={tmp2.name}", path
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return tmp2.name
    except Exception as e:
        logging.warning(f"Ghostscript falhou em '{path}': {e}")
        try:
            os.remove(tmp2.name)
        except Exception:
            pass

    # fallback: retorna original
    return path

def move_to_processed(path: str, root_dir: str) -> None:
    """Move arquivo processado para a subpasta 'Processado'."""
    dest_dir = os.path.join(root_dir, "Processado")
    try:
        os.makedirs(dest_dir, exist_ok=True)
        base = os.path.basename(path)
        dest = os.path.join(dest_dir, base)
        if os.path.exists(dest):
            name, ext = os.path.splitext(base)
            idx = 1
            new_dest = os.path.join(dest_dir, f"{name}_{idx}{ext}")
            while os.path.exists(new_dest):
                idx += 1
                new_dest = os.path.join(dest_dir, f"{name}_{idx}{ext}")
            dest = new_dest
        os.replace(path, dest)
    except Exception as e:
        logging.error(f"Falha ao mover '{path}' para '{dest_dir}': {e}")
