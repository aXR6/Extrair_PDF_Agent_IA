# extractors.py
import logging
import subprocess
import tempfile
import shutil
import os

import fitz
import pdfplumber
import pytesseract
from pdfminer.high_level import extract_text as pdfminer_extract_text
from langchain_community.document_loaders import (
    PyPDFLoader,
    PDFMinerLoader,
    UnstructuredWordDocumentLoader
)
from PIL import Image

from config import (
    OCR_LANGUAGES,
    OCR_THRESHOLD,
    PDF2IMAGE_TIMEOUT,
    TESSERACT_CONFIG,
)
from utils import repair_pdf
from constants import VALID_EXTS, IMAGE_EXTS


# ---------------------------------------------------------------------------
# Estratégias individuais
# ---------------------------------------------------------------------------
class PyPDFStrategy:
    def extract(self, path: str) -> str:
        docs = PyPDFLoader(path).load()
        return "\n".join(page.page_content for page in docs)


class PDFMinerStrategy:
    def extract(self, path: str) -> str:
        docs = PDFMinerLoader(path).load()
        return "\n".join(page.page_content for page in docs)


class PDFMinerLowLevelStrategy:
    def extract(self, path: str) -> str:
        try:
            return pdfminer_extract_text(path)
        except Exception as e:
            logging.error(f"Erro PDFMiner low-level: {e}")
            return ""


class UnstructuredStrategy:
    def extract(self, path: str) -> str:
        docs = UnstructuredWordDocumentLoader(path).load()
        return "\n".join(page.page_content for page in docs)


class OCRStrategy:
    def __init__(self, threshold: int = OCR_THRESHOLD):
        self.threshold = threshold

    def extract(self, path: str) -> str:
        try:
            # Tenta extrair texto diretamente
            doc = fitz.open(path)
            raw = "\n".join(page.get_text() for page in doc)
            doc.close()
            if len(raw.strip()) > self.threshold:
                return raw

            # Caso contrário, usa OCR em imagem
            from pdf2image import convert_from_path
            images = convert_from_path(
                path, dpi=300, timeout=PDF2IMAGE_TIMEOUT
            )
            return "\n\n".join(
                pytesseract.image_to_string(
                    img, lang=OCR_LANGUAGES, config=TESSERACT_CONFIG
                )
                for img in images
            )
        except Exception as e:
            logging.error(f"Erro OCRStrategy: {e}")
            return ""


class PDFPlumberStrategy:
    def extract(self, path: str) -> str:
        texts = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                texts.append(page.extract_text() or "")
        return "\n".join(texts)




class PyMuPDF4LLMStrategy:
    def extract(self, path: str) -> str:
        try:
            import pymupdf4llm
            return pymupdf4llm.to_markdown(path)
        except Exception as e:
            logging.error(f"Erro PyMuPDF4LLMStrategy: {e}")
            return ""


class ImageOCRStrategy:
    def extract(self, path: str) -> str:
        try:
            img = Image.open(path)
            return pytesseract.image_to_string(
                img, lang=OCR_LANGUAGES, config=TESSERACT_CONFIG
            )
        except Exception as e:
            logging.error(f"Erro ImageOCRStrategy: {e}")
            return ""


# ---------------------------------------------------------------------------
# Mapa de estratégias
# ---------------------------------------------------------------------------
STRATEGIES_MAP = {
    "pypdf": PyPDFStrategy(),
    "pdfminer": PDFMinerStrategy(),
    "pdfminer-low": PDFMinerLowLevelStrategy(),
    "unstructured": UnstructuredStrategy(),
    "ocr": OCRStrategy(),
    "plumber": PDFPlumberStrategy(),
    "pymupdf4llm": PyMuPDF4LLMStrategy(),
    "image": ImageOCRStrategy(),
}


def extract_text(path: str, strategy: str) -> str:
    """
    Extrai texto de arquivos PDF, DOCX e imagens com fallbacks:
      - IMG: usa ImageOCRStrategy
      - DOCX: sempre usa UnstructuredStrategy
      - PDF: tenta repair_pdf, estratégia primária, e cascata de fallbacks
    """
    lower = path.lower()
    # Imagens --> OCR direto
    if lower.endswith(IMAGE_EXTS):
        return STRATEGIES_MAP["image"].extract(path)

    # DOCX --> Unstructured
    if lower.endswith(".docx"):
        return STRATEGIES_MAP["unstructured"].extract(path)

    # PDF --> reparar antes
    repaired = repair_pdf(path)
    try:
        # 1) Estratégia primária
        loader = STRATEGIES_MAP.get(strategy)
        text = ""
        if loader:
            try:
                text = loader.extract(repaired)
            except Exception as e:
                logging.warning(f"Loader '{strategy}' falhou: {e}")
        else:
            logging.error(f"Estratégia desconhecida: {strategy}")

        if len(text.strip()) > OCR_THRESHOLD:
            return text

        # 2) Fallbacks para PDF
        try:
            txt = pdfminer_extract_text(repaired)
            if len(txt.strip()) > OCR_THRESHOLD:
                return txt
        except Exception:
            pass

        try:
            with pdfplumber.open(repaired) as pdf:
                txt = "\n".join(page.extract_text() or "" for page in pdf.pages)
            if len(txt.strip()) > OCR_THRESHOLD:
                return txt
        except Exception:
            pass

        if shutil.which("pdftotext"):
            tmp = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
            tmp.close()
            try:
                subprocess.run([
                    "pdftotext", "-layout", repaired, tmp.name
                ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                with open(tmp.name, encoding="utf-8", errors="ignore") as fh:
                    txt = fh.read()
                if len(txt.strip()) > OCR_THRESHOLD:
                    return txt
            except Exception:
                pass
            finally:
                try:
                    os.remove(tmp.name)
                except Exception:
                    pass
        else:
            try:
                import pdftotext
                with open(repaired, "rb") as f:
                    pdf_doc = pdftotext.PDF(f)
                txt = "\n\n".join(pdf_doc)
                if len(txt.strip()) > OCR_THRESHOLD:
                    return txt
            except Exception:
                pass

        try:
            from pdf2image import convert_from_path
            images = convert_from_path(
                repaired, dpi=300, timeout=PDF2IMAGE_TIMEOUT
            )
            return "\n\n".join(
                pytesseract.image_to_string(
                    img, lang=OCR_LANGUAGES, config=TESSERACT_CONFIG
                )
                for img in images
            )
        except Exception as e:
            logging.error(f"OCR final falhou: {e}")
            return text
    finally:
        if repaired != path:
            try:
                os.remove(repaired)
            except Exception:
                pass
