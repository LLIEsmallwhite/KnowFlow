"""
文档服务

负责文档的完整处理流水线：
1. 文件上传 → MinIO 存储
2. 文档解析 → 文本提取（异步 Celery 任务）
3. 文本分块 → 自适应三层分块
4. 向量化 → 写入 Milvus + 更新 Chunk 记录

处理状态追踪通过 Document.status 字段实现。
"""

import uuid
import logging
from typing import List, Optional
from pathlib import Path

from app.utils.chunking import AdaptiveChunker, Chunk, ChunkingConfig
from app.utils.document_loaders import load_document, Document
from app.utils.text_processing import build_content_signature

logger = logging.getLogger(__name__)


class DocumentService:
    """
    文档处理服务

    负责文档的解析、分块和索引编排。
    实际的向量化和 Milvus 写入由检索模块负责。
    """

    # 支持的文件类型
    SUPPORTED_TYPES = {
        "pdf", "docx", "doc", "md", "markdown",
        "html", "htm", "txt", "text",
    }

    def __init__(self):
        """初始化文档服务"""
        pass

    def is_supported(self, file_type: str) -> bool:
        """检查文件类型是否支持"""
        return file_type.lower() in self.SUPPORTED_TYPES

    def parse_document(
        self, file_path: str, file_type: Optional[str] = None
    ) -> Document:
        """
        解析文档为纯文本

        Args:
            file_path: 文件路径（本地路径）
            file_type: 文件类型

        Returns:
            解析后的 Document 对象，包含 content 和 metadata

        Raises:
            ValueError: 不支持的文件类型
            Exception: 解析失败
        """
        logger.info(f"Parsing document: {file_path}, type={file_type}")
        try:
            doc = load_document(file_path, file_type)
            logger.info(f"Parsed: {len(doc.content)} chars, title={doc.metadata.get('title')}")
            return doc
        except Exception as e:
            logger.error(f"Document parsing failed: {e}")
            raise

    def chunk_document(
        self,
        content: str,
        chunk_size: int = 512,
        chunk_overlap: int = 80,
        enable_parent_child: bool = True,
        parent_chunk_size: int = 4096,
        child_chunk_size: int = 384,
    ) -> List[Chunk]:
        """
        对文档文本进行自适应分块

        Args:
            content: 文档文本
            chunk_size: 目标 Chunk 大小
            chunk_overlap: Chunk 重叠
            enable_parent_child: 是否启用 Parent-Child
            parent_chunk_size: 父块大小
            child_chunk_size: 子块大小

        Returns:
            Chunk 列表
        """
        logger.info(
            f"Chunking document: size={len(content)} chars, "
            f"chunk_size={chunk_size}, parent_child={enable_parent_child}"
        )

        config = ChunkingConfig(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            enable_parent_child=enable_parent_child,
            parent_chunk_size=parent_chunk_size,
            child_chunk_size=child_chunk_size,
        )
        chunker = AdaptiveChunker(config)
        chunks = chunker.chunk(content)

        # 为每个 Chunk 生成内容签名（供后续去重使用）
        for chunk in chunks:
            chunk.metadata["content_hash"] = build_content_signature(chunk.content)

        logger.info(f"Generated {len(chunks)} chunks")
        return chunks

    def process_document_pipeline(
        self,
        file_path: str,
        file_type: Optional[str] = None,
        chunking_config: Optional[ChunkingConfig] = None,
    ) -> tuple[Document, List[Chunk]]:
        """
        完整文档处理流水线：解析 + 分块

        Args:
            file_path: 文件路径
            file_type: 文件类型
            chunking_config: 分块配置（None 使用默认）

        Returns:
            (Document, List[Chunk]) 元组
        """

        # Step 1: 解析文档
        doc = self.parse_document(file_path, file_type)

        # Step 2: 分块
        if chunking_config is None:
            chunking_config = ChunkingConfig()
        chunks = self.chunk_document(
            doc.content,
            chunk_size=chunking_config.chunk_size,
            chunk_overlap=chunking_config.chunk_overlap,
            enable_parent_child=chunking_config.enable_parent_child,
            parent_chunk_size=chunking_config.parent_chunk_size,
            child_chunk_size=chunking_config.child_chunk_size,
        )

        logger.info(
            f"Document pipeline complete: {file_path} → "
            f"{len(doc.content)} chars → {len(chunks)} chunks"
        )
        return doc, chunks


# ─── 全局单例 ───
doc_service = DocumentService()
