#extractors.py
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
from config import OCR_LANGUAGES
import pymupdf4llm  # Nova dependência para extração em Markdown

# ---------------------------------------------------------------------------
# EXTRACTION STRATEGIES
# ---------------------------------------------------------------------------

def is_extraction_allowed(path: str) -> bool:
    """Verifica se o PDF permite extração direta de texto (não criptografado)."""
    try:
        reader = PdfReader(path)
        if reader.is_encrypted:
            logging.warning(f"PDF criptografado: {path}")
            return False
        return True
    except Exception as e:
        logging.error(f"Erro ao verificar permissão: {e}")
        return False

def fallback_ocr(path: str, threshold: int = 100) -> str:
    """Tenta extrair texto digital e cai para OCR se estiver escaneado ou vazio."""
    try:
        doc = fitz.open(path)
        raw = "\n".join(page.get_text() for page in doc)
        doc.close()
        if len(raw.strip()) > threshold:
            return raw
        images = convert_from_path(path, dpi=300)
        return "\n\n".join(
            pytesseract.image_to_string(img, lang=OCR_LANGUAGES) for img in images
        )
    except Exception as e:
        logging.error(f"Erro no OCR fallback: {e}")
        return ""

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
    def __init__(self, threshold: int = 100):
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
            # Converte PDF em Markdown compatível com GitHub
            return pymupdf4llm.to_markdown(path)
        except Exception as e:
            logging.error(f"Erro no PyMuPDF4LLMStrategy: {e}")
            return ""

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