import logging
import subprocess
import tempfile

import fitz
import pdfplumber
import pytesseract
from pdfminer.high_level import extract_text as pdfminer_extract_text
from tika import parser
from langchain_community.document_loaders import (
    PyPDFLoader,
    PDFMinerLoader,
    UnstructuredWordDocumentLoader
)

from config import OCR_LANGUAGES, OCR_THRESHOLD
from utils import repair_pdf

# ---------------------------------------------------------------------------
# Estratégias de extração (usadas pelo extract_text)
# ---------------------------------------------------------------------------
class PyPDFStrategy:
    def extract(self, path: str) -> str:
        loader = PyPDFLoader(path)
        docs = loader.load()
        return "\n".join(d.page_content for d in docs)

class PDFMinerStrategy:
    def extract(self, path: str) -> str:
        loader = PDFMinerLoader(path)
        docs = loader.load()
        return "\n".join(d.page_content for d in docs)

class PDFMinerLowLevelStrategy:
    def extract(self, path: str) -> str:
        try:
            return pdfminer_extract_text(path)
        except Exception as e:
            logging.error(f"Erro no PDFMiner low-level: {e}")
            return ""

class UnstructuredStrategy:
    def extract(self, path: str) -> str:
        loader = UnstructuredWordDocumentLoader(path)
        docs = loader.load()
        return "\n".join(d.page_content for d in docs)

class OCRStrategy:
    def __init__(self, threshold: int = OCR_THRESHOLD):
        self.threshold = threshold
    def extract(self, path: str) -> str:
        try:
            doc = fitz.open(path)
            raw = "\n".join(page.get_text() for page in doc)
            doc.close()
            if len(raw.strip()) > self.threshold:
                return raw
            images = convert_from_path(path, dpi=300)
            return "\n\n".join(
                pytesseract.image_to_string(img, lang=OCR_LANGUAGES)
                for img in images
            )
        except Exception as e:
            logging.error(f"Erro no OCRStrategy: {e}")
            return ""

class PDFPlumberStrategy:
    def extract(self, path: str) -> str:
        text = []
        with pdfplumber.open(path) as pdf:
            for p in pdf.pages:
                t = p.extract_text() or ""
                text.append(t)
        return "\n".join(text)

class TikaStrategy:
    def extract(self, path: str) -> str:
        parsed = parser.from_file(path)
        return parsed.get("content", "") or ""

class PyMuPDF4LLMStrategy:
    def extract(self, path: str) -> str:
        try:
            import pymupdf4llm
            return pymupdf4llm.to_markdown(path)
        except Exception as e:
            logging.error(f"Erro no PyMuPDF4LLMStrategy: {e}")
            return ""

STRATEGIES_MAP = {
    "pypdf": PyPDFStrategy(),
    "pdfminer": PDFMinerStrategy(),
    "pdfminer-low": PDFMinerLowLevelStrategy(),
    "unstructured": UnstructuredStrategy(),
    "ocr": OCRStrategy(),
    "plumber": PDFPlumberStrategy(),
    "tika": TikaStrategy(),
    "pymupdf4llm": PyMuPDF4LLMStrategy()
}


def extract_text(path: str, strategy: str) -> str:
    """
    1) Repara o PDF (mutool → pikepdf → gs)
    2) Extrai pela estratégia escolhida
    3) Fallbacks: PDFMiner → Tika → PDFPlumber → pdftotext → OCR
    """
    # 1) reparo
    path = repair_pdf(path)

    # 2) tentativa primária
    text = ""
    loader = STRATEGIES_MAP.get(strategy)
    if loader:
        try:
            text = loader.extract(path)
        except Exception as e:
            logging.warning(f"Loader '{strategy}' falhou: {e}")
    else:
        logging.error(f"Estratégia desconhecida: {strategy}")

    if len(text.strip()) > OCR_THRESHOLD:
        return text

    # 3) fallbacks em cascata

    # PDFMiner low-level
    try:
        txt = pdfminer_extract_text(path)
        if len(txt.strip()) > OCR_THRESHOLD:
            return txt
    except Exception:
        pass

    # Tika
    try:
        parsed = parser.from_file(path)
        txt = parsed.get("content", "") or ""
        if len(txt.strip()) > OCR_THRESHOLD:
            return txt
    except Exception:
        pass

    # PDFPlumber
    try:
        with pdfplumber.open(path) as pdf:
            txt = "\n".join(p.extract_text() or "" for p in pdf.pages)
        if len(txt.strip()) > OCR_THRESHOLD:
            return txt
    except Exception:
        pass

    # pdftotext (poppler-utils)
    try:
        tmp = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
        subprocess.run(
            ["pdftotext", "-layout", path, tmp.name],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        txt = open(tmp.name, encoding="utf-8", errors="ignore").read()
        if len(txt.strip()) > OCR_THRESHOLD:
            return txt
    except Exception:
        pass

    # OCR final
    try:
        from pdf2image import convert_from_path
        images = convert_from_path(path, dpi=300)
        return "\n\n".join(
            pytesseract.image_to_string(img, lang=OCR_LANGUAGES)
            for img in images
        )
    except Exception as e:
        logging.error(f"OCR fallback também falhou: {e}")
        return text