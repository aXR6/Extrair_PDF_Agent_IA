import fitz
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract
import pdfplumber
from tika import parser
import logging
from pdfminer.high_level import extract_text as pdfminer_extract_text
from langchain_community.document_loaders import (
    PyPDFLoader,
    PDFMinerLoader,
    UnstructuredWordDocumentLoader
)
from config import OCR_LANGUAGES, OCR_THRESHOLD
import pymupdf4llm  # Nova dependência para extração em Markdown

# ---------------------------------------------------------------------------
# Extraction Strategies
# ---------------------------------------------------------------------------

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
                pytesseract.image_to_string(img, lang=OCR_LANGUAGES) for img in images
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

# Mapeamento de estratégias
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

# ---------------------------------------------------------------------------
# Fallback OCR Multi-camada
# ---------------------------------------------------------------------------

def fallback_ocr(path: str, threshold: int = OCR_THRESHOLD) -> str:
    """
    Fluxo de extração robusto:
      1) PyMuPDF
      2) PDFMiner low-level
      3) Apache Tika
      4) PDFPlumber
      5) OCR (pytesseract + pdf2image)
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

    # 2) PDFMiner low-level
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
            pages = [p.extract_text() or "" for p in pdf.pages]
        plumber_text = "\n".join(pages)
        if len(plumber_text.strip()) > threshold:
            return plumber_text
    except Exception:
        logging.debug("PDFPlumber falhou, tentando OCR…")

    # 5) OCR final
    try:
        images = convert_from_path(path, dpi=300)
        return "\n\n".join(
            pytesseract.image_to_string(img, lang=OCR_LANGUAGES)
            for img in images
        )
    except Exception as e:
        logging.error(f"OCR fallback também falhou: {e}")
        return text

# ---------------------------------------------------------------------------
# Função unificada de extração
# ---------------------------------------------------------------------------

def extract_text(path: str, strategy: str) -> str:
    """
    Extrai texto usando a estratégia solicitada e, se necessário,
    aplica fallback multi-camada.
    """
    text = ""
    loader = STRATEGIES_MAP.get(strategy)
    if loader:
        try:
            text = loader.extract(path)
        except Exception as e:
            logging.warning(f"Loader '{strategy}' falhou em {path}: {e}")
    else:
        logging.error(f"Estratégia desconhecida: {strategy}")

    if not text or len(text.strip()) < OCR_THRESHOLD:
        logging.info(f"Aplicando fallback geral em {path}")
        text = fallback_ocr(path, threshold=OCR_THRESHOLD)

    return text

# ---------------------------------------------------------------------------
# METADATA BUILDER
# ---------------------------------------------------------------------------

def build_record(path: str, text: str) -> dict:
    """Extrai metadados básicos via PyMuPDF e PyPDF2."""
    try:
        doc = fitz.open(path)
        info = doc.metadata.copy() or {}
        info['numpages'] = doc.page_count
        doc.close()
    except Exception as e:
        logging.error(f"Erro ao extrair metadados: {e}")
        info = {}
    return {'text': text, 'info': info, 'version': "2.16.105"}