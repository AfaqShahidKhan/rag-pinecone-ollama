from .pdf_loader import PdfDocumentLoader
from .docx_loader import DocxDocumentLoader
from .pptx_loader import PptxDocumentLoader
from .html_loader import HtmlLoader
from .json_loader import JsonLoader
from .ocr_loader import OcrLoader

__all__ = [
    "PdfDocumentLoader",
    "DocxDocumentLoader",
    "PptxDocumentLoader",
    "HtmlLoader",
    "JsonLoader",
    "OcrLoader",
]