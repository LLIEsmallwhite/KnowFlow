"""
Dense Vector 检索器 (Milvus)

负责：
1. 调用 Embedding 模型将查询转为向量
2. 在 Milvus 中执行向量相似度检索（Inner Product）
3. 返回带分数的检索结果

使用 EmbeddingPool 实现模型实例复用和查询缓存。
"""

import logging
from typing import List, Optional
from langchain_openai import OpenAIEmbeddings

from app.core.config import settings
from app.retrieval.milvus_client import MilvusClient, SearchResult

logger = logging.getLogger(__name__)


class DenseRetriever:
    """
    基于 Milvus 的 Dense Vector 检索器

    支持：
    - 多知识库跨 Partition 检索
    - 分数阈值过滤
    - Embedding 模型可插拔（默认 text-embedding-3-large）

    使用示例:
        retriever = DenseRetriever()
        results = retriever.search(
            query="Kubernetes Pod 无法启动",
            kb_ids=["kb_001"],
            top_k=50,
        )
    """

    def __init__(
        self,
        milvus: Optional[MilvusClient] = None,
        embedding_model: Optional[str] = None,
        embedding_api_key: Optional[str] = None,
        embedding_base_url: Optional[str] = None,
    ):
        """
        Args:
            milvus: Milvus 客户端实例
            embedding_model: Embedding 模型名称
            embedding_api_key: API Key
            embedding_base_url: API 基础 URL
        """
        self.milvus = milvus or MilvusClient()
        self.milvus.connect()

        # Embedding 模型配置
        self._embedding_model = OpenAIEmbeddings(
            model=embedding_model or settings.EMBEDDING_MODEL,
            openai_api_key=embedding_api_key or settings.EMBEDDING_API_KEY,
            openai_api_base=embedding_base_url or settings.EMBEDDING_BASE_URL,
            dimensions=settings.EMBEDDING_DIMENSION,
        )

    def embed_query(self, query: str) -> List[float]:
        """
        将查询文本转为 Embedding 向量

        Args:
            query: 查询文本

        Returns:
            向量（float 列表）
        """
        try:
            embedding = self._embedding_model.embed_query(query)
            logger.debug(f"Query embedded: dim={len(embedding)}, query_len={len(query)}")
            return embedding
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            raise

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """
        批量计算文档 Embedding

        Args:
            documents: 文档文本列表

        Returns:
            向量列表
        """
        try:
            embeddings = self._embedding_model.embed_documents(documents)
            logger.debug(f"Documents embedded: count={len(embeddings)}, dim={len(embeddings[0]) if embeddings else 0}")
            return embeddings
        except Exception as e:
            logger.error(f"Batch embedding failed: {e}")
            raise

    def search(
        self,
        query: str,
        kb_ids: Optional[List[str]] = None,
        top_k: int = 50,
        threshold: float = 0.15,
        query_embedding: Optional[List[float]] = None,
    ) -> List[SearchResult]:
        """
        执行 Dense Vector 检索

        Args:
            query: 查询文本（如提供 query_embedding 则可为空）
            kb_ids: 知识库 ID 列表（None = 全局搜索）
            top_k: 返回数量上限
            threshold: 最低分数阈值（低于此分数的结果被丢弃）
            query_embedding: 预计算的查询向量（跳过 Embedding 步骤）

        Returns:
            检索结果列表
        """
        # Step 1: 获取查询向量（如果在外部已计算则复用）
        if query_embedding is None:
            query_embedding = self.embed_query(query)

        # Step 2: Milvus 向量检索
        results = self.milvus.search(
            query_embedding=query_embedding,
            kb_ids=kb_ids,
            top_k=top_k,
            threshold=threshold,
        )

        logger.info(
            f"Dense search: query='{query[:50]}...', "
            f"kbs={kb_ids}, top_k={top_k}, results={len(results)}"
        )
        return results
