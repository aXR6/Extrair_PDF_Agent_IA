import tempfile
import subprocess
import logging

import fitz
import pytesseract
import pdfplumber
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
from pdfminer.high_level import extract_text as pdfminer_extract_text
from tika import parser
from langchain_community.document_loaders import (
    PyPDFLoader,
    PDFMinerLoader,
    UnstructuredWordDocumentLoader
)
from config import OCR_LANGUAGES, OCR_THRESHOLD
import pymupdf4llm  # Nova dependência para extração em Markdown

# ---------------------------------------------------------------------------
# Função Unificada de Extração + Fallback Multi-camada
# ---------------------------------------------------------------------------

# Mapeamento de estratégias (usado na função extract_text)
class PyPDFStrategy:
    """Extrai com loader PyPDFLoader do LangChain."""
    def extract(self, path: str) -> str:
        loader = PyPDFLoader(path)
        docs = loader.load()
        return "\n".join(d.page_content for d in docs)

class PDFMinerStrategy:
    """Extrai com loader PDFMinerLoader do LangChain."""
    def extract(self, path: str) -> str:
        loader = PDFMinerLoader(path)
        docs = loader.load()
        return "\n".join(d.page_content for d in docs)

class PDFMinerLowLevelStrategy:
    """Extrai usando pdfminer.six de baixo nível (extract_text)."""
    def extract(self, path: str) -> str:
        try:
            return pdfminer_extract_text(path)
        except Exception as e:
            logging.error(f"Erro no PDFMiner low-level: {e}")
            return ""

class UnstructuredStrategy:
    """Extrai documentos .docx com UnstructuredWordDocumentLoader."""
    def extract(self, path: str) -> str:
        loader = UnstructuredWordDocumentLoader(path)
        docs = loader.load()
        return "\n".join(d.page_content for d in docs)

class OCRStrategy:
    """Extrai texto e cai para OCR se híbrido com threshold."""
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
    """Extrai texto e tabelas usando PDFPlumber."""
    def extract(self, path: str) -> str:
        text = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
        return "\n".join(text)

class TikaStrategy:
    """Extrai conteúdo via Apache Tika."""
    def extract(self, path: str) -> str:
        parsed = parser.from_file(path)
        return parsed.get("content", "") or ""

class PyMuPDF4LLMStrategy:
    """Extrai PDF em formato Markdown usando PyMuPDF4LLM."""
    def extract(self, path: str) -> str:
        try:
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
    Extrai texto usando a estratégia inicial e aplica fallback robusto se necessário.
    """
    text = ""
    loader = STRATEGIES_MAP.get(strategy)
    if loader:
        try:
            text = loader.extract(path)
        except Exception as e:
            logging.warning(f"Loader '{strategy}' falhou: {e}")
    else:
        logging.error(f"Estratégia desconhecida: {strategy}")

    if not text or len(text.strip()) < OCR_THRESHOLD:
        logging.info(f"Aplicando fallback OCR robusto em {path}")
        text = fallback_ocr(path, threshold=OCR_THRESHOLD)
    return text


def fallback_ocr(path: str, threshold: int = OCR_THRESHOLD) -> str:
    """
    Fluxo de extração robusto:
      1) PyMuPDF
      2) PDFMiner low-level
      3) Apache Tika
      4) PDFPlumber
      5) Ghostscript recompactação
      6) pdftotext (poppler-utils)
      7) OCR final (pytesseract)
    """
    text = ""

    # 1) PyMuPDF
    try:
        doc = fitz.open(path)
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        if len(text.strip()) > threshold:
            return text
    except Exception:
        logging.debug("PyMuPDF falhou, tentando PDFMiner…")

    # 2) PDFMiner
    try:
        text = pdfminer_extract_text(path)
        if len(text.strip()) > threshold:
            return text
    except Exception:
        logging.debug("PDFMiner falhou, tentando Tika…")

    # 3) Apache Tika
    try:
        parsed = parser.from_file(path)
        tika_text = parsed.get("content", "") or ""
        if len(tika_text.strip()) > threshold:
            return tika_text
    except Exception as e:
        logging.warning(f"Tika falhou: {e}")

    # 4) PDFPlumber
    try:
        with pdfplumber.open(path) as pdf:
            plumber_text = "\n".join(p.extract_text() or "" for p in pdf.pages)
        if len(plumber_text.strip()) > threshold:
            return plumber_text
    except Exception:
        logging.debug("PDFPlumber falhou, tentando Ghostscript…")

    # 5) Ghostscript recompactação
    try:
        tmp_pdf = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        cmd = [
            "gs", "-q", "-dNOPAUSE", "-dBATCH",
            "-sDEVICE=pdfwrite", "-dCompatibilityLevel=1.4",
            "-dPDFSETTINGS=/prepress",
            f"-sOutputFile={tmp_pdf.name}", path
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        path = tmp_pdf.name
        # tenta novamente
        doc = fitz.open(path)
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        if len(text.strip()) > threshold:
            return text
    except Exception:
        logging.debug("Ghostscript falhou, tentando pdftotext…")

    # 6) pdftotext (poppler-utils)
    try:
        tmp_txt = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
        subprocess.run(
            ["pdftotext", "-layout", path, tmp_txt.name],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        pdftxt = open(tmp_txt.name, encoding="utf-8", errors="ignore").read()
        if len(pdftxt.strip()) > threshold:
            return pdftxt
    except Exception as e:
        logging.debug(f"pdftotext falhou: {e}")

    # 7) OCR final
    try:
        images = convert_from_path(path, dpi=300)
        return "\n\n".join(
            pytesseract.image_to_string(img, lang=OCR_LANGUAGES)
            for img in images
        )
    except Exception as e:
        logging.error(f"OCR fallback também falhou: {e}")
        return text