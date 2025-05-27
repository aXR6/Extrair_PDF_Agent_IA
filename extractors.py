import logging
import subprocess
import tempfile

import fitz
import pdfplumber
import pytesseract
from PyPDF2 import PdfReader
from pdfminer.high_level import extract_text as pdfminer_extract_text
from tika import parser
from langchain_community.document_loaders import (
    PyPDFLoader,
    PDFMinerLoader,
    UnstructuredWordDocumentLoader
)

from config import OCR_LANGUAGES
from utils import repair_pdf

def extract_text(path: str, strategy: str, threshold: int = 100) -> str:
    """
    Extrai texto em pipeline unificado:
      a) Repair via mutool/pikepdf/ghostscript (repair_pdf)
      b) Estratégia escolhida (PyPDF2, PDFMiner, ...)
      c) Fallback OCR robusto:
         - pdfminer.low-level → Tika → PDFPlumber
         - pdftotext (Poppler) :contentReference[oaicite:5]{index=5}
         - OCR (pytesseract)
    """
    # a) tenta reparo
    path = repair_pdf(path)

    text = ""
    # b) tentativa primária
    try:
        if strategy == "pypdf":
            docs = PyPDFLoader(path).load()
            text = "\n".join(d.page_content for d in docs)
        elif strategy == "pdfminer":
            docs = PDFMinerLoader(path).load()
            text = "\n".join(d.page_content for d in docs)
        elif strategy == "pdfminer-low":
            text = pdfminer_extract_text(path)
        elif strategy == "unstructured":
            docs = UnstructuredWordDocumentLoader(path).load()
            text = "\n".join(d.page_content for d in docs)
        elif strategy == "plumber":
            with pdfplumber.open(path) as pdf:
                text = "\n".join(p.extract_text() or "" for p in pdf.pages)
        elif strategy == "tika":
            parsed = parser.from_file(path)
            text = parsed.get("content", "") or ""
        else:  # pymupdf4llm ou ocr puro
            from extractors import PyMuPDF4LLMStrategy, OCRStrategy
            if strategy == "pymupdf4llm":
                text = PyMuPDF4LLMStrategy().extract(path)
            else:
                text = OCRStrategy(threshold).extract(path)
    except Exception as e:
        logging.warning(f"Erro na estratégia '{strategy}': {e}")

    if len(text.strip()) > threshold:
        return text

    # c) Fallback avançado
    # 1) pdfminer low-level
    try:
        text = pdfminer_extract_text(path)
        if len(text.strip()) > threshold:
            return text
    except Exception:
        pass

    # 2) Tika
    try:
        parsed = parser.from_file(path)
        tika_txt = parsed.get("content", "") or ""
        if len(tika_txt.strip()) > threshold:
            return tika_txt
    except Exception:
        pass

    # 3) PDFPlumber
    try:
        with pdfplumber.open(path) as pdf:
            plumber_txt = "\n".join(p.extract_text() or "" for p in pdf.pages)
        if len(plumber_txt.strip()) > threshold:
            return plumber_txt
    except Exception:
        pass

    # 4) pdftotext (Poppler) :contentReference[oaicite:6]{index=6}
    try:
        tmp = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
        subprocess.run(
            ["pdftotext", "-layout", path, tmp.name],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        txt = open(tmp.name, encoding="utf-8", errors="ignore").read()
        if len(txt.strip()) > threshold:
            return txt
    except Exception as e:
        logging.debug(f"pdftotext falhou: {e}")

    # 5) OCR final :contentReference[oaicite:7]{index=7}
    try:
        images = fitz.open(path)
        pages = [page.get_pixmap(dpi=300).pil_tobytes() for page in images]
        return "\n\n".join(
            pytesseract.image_to_string(img, lang=OCR_LANGUAGES)
            for img in pages
        )
    except Exception as e:
        logging.error(f"OCR fallback também falhou: {e}")
        return text