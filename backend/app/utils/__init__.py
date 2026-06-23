"""
Utility module — Document loaders, chunking, text processing.
"""

from app.utils.document_loaders import (
    Document,
    BaseLoader,
    PDFLoader,
    DocxLoader,
    MarkdownLoader,
    HTMLLoader,
    TxtLoader,
    CSVLoader,
    ExcelLoader,
    PPTLoader,
    LOADER_MAP,
    load_document,
)

from app.utils.chunking import AdaptiveChunker, Chunk, ChunkingConfig
from app.utils.text_processing import normalize_content, build_content_signature

__all__ = [
    "Document", "BaseLoader",
    "PDFLoader", "DocxLoader", "MarkdownLoader", "HTMLLoader",
    "TxtLoader", "CSVLoader", "ExcelLoader", "PPTLoader",
    "LOADER_MAP", "load_document",
    "AdaptiveChunker", "Chunk", "ChunkingConfig",
    "normalize_content", "build_content_signature",
]
