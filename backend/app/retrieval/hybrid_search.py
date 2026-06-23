"""
混合检索编排器

并行执行 Dense Vector 检索 + BM25 关键词检索，
为后续的动态 RRF 融合提供原始检索结果。

设计原则：
- 向量检索和 BM25 检索完全并行（ThreadPoolExecutor）
- 单个检索器失败不影响另一个（优雅降级）
- 结果保持原始分数，不做任何归一化（由 RRF 模块负责融合）
"""

import logging
import concurrent.futures
from typing import List, Dict, Optional
from dataclasses import dataclass, field

from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.bm25_retriever import BM25Retriever, BM25Result
from app.retrieval.milvus_client import SearchResult as DenseResult

logger = logging.getLogger(__name__)


@dataclass
class HybridSearchResult:
    """
    混合检索的原始结果集合

    包含向量检索和关键词检索两路结果，保持各自的原始分数。
    后续由 DynamicRRF 模块进行融合。
    """
    query: str
    vector_results: List[DenseResult] = field(default_factory=list)
    keyword_results: List[BM25Result] = field(default_factory=list)
    vector_error: Optional[str] = None     # 向量检索错误信息
    keyword_error: Optional[str] = None    # 关键词检索错误信息

    @property
    def total_count(self) -> int:
        """两路检索结果总数"""
        return len(self.vector_results) + len(self.keyword_results)

    @property
    def has_results(self) -> bool:
        """是否有任何检索结果"""
        return len(self.vector_results) > 0 or len(self.keyword_results) > 0


class HybridSearchOrchestrator:
    """
    混合检索编排器

    并行调度 Dense + BM25 检索，对上层提供统一接口。

    使用示例:
        orchestrator = HybridSearchOrchestrator(dense_retriever, bm25_retriever)
        result = orchestrator.search(
            query="Kubernetes Pod 无法启动如何排查",
            kb_ids=["kb_001"],
            top_k=50,
        )
    """

    def __init__(
        self,
        dense_retriever: Optional[DenseRetriever] = None,
        bm25_retriever: Optional[BM25Retriever] = None,
    ):
        """
        Args:
            dense_retriever: Dense Vector 检索器
            bm25_retriever: BM25 关键词检索器
        """
        self.dense = dense_retriever or DenseRetriever()
        self.bm25 = bm25_retriever or BM25Retriever()

    def search(
        self,
        query: str,
        kb_ids: Optional[List[str]] = None,
        vector_top_k: int = 50,
        keyword_top_k: int = 50,
        vector_threshold: float = 0.15,
        keyword_threshold: float = 0.30,
    ) -> HybridSearchResult:
        """
        执行混合检索（并行）

        Args:
            query: 查询文本（建议使用 rewritten_query）
            kb_ids: 知识库 ID 列表
            vector_top_k: 向量检索返回数
            keyword_top_k: 关键词检索返回数
            vector_threshold: 向量检索分数阈值
            keyword_threshold: 关键词检索分数阈值

        Returns:
            HybridSearchResult（包含两路检索结果）
        """
        result = HybridSearchResult(query=query)

        # 并行执行两路检索
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            # 提交向量检索
            vector_future = executor.submit(
                self._safe_vector_search,
                query, kb_ids, vector_top_k, vector_threshold,
            )

            # 提交关键词检索
            keyword_future = executor.submit(
                self._safe_keyword_search,
                query, kb_ids, keyword_top_k, keyword_threshold,
            )

            # 获取结果（各自失败不影响对方）
            result.vector_results, result.vector_error = vector_future.result()
            result.keyword_results, result.keyword_error = keyword_future.result()

        # 日志汇总
        logger.info(
            f"Hybrid search complete: query='{query[:50]}...', "
            f"vector={len(result.vector_results)}"
            f"{' (ERROR)' if result.vector_error else ''}, "
            f"keyword={len(result.keyword_results)}"
            f"{' (ERROR)' if result.keyword_error else ''}"
        )

        return result

    def _safe_vector_search(
        self, query: str, kb_ids: List[str], top_k: int, threshold: float,
    ) -> tuple:
        """安全执行向量检索（捕获异常）"""
        try:
            results = self.dense.search(
                query=query,
                kb_ids=kb_ids,
                top_k=top_k,
                threshold=threshold,
            )
            return results, None
        except Exception as e:
            logger.error(f"Vector search failed: {e}", exc_info=True)
            return [], str(e)

    def _safe_keyword_search(
        self, query: str, kb_ids: List[str], top_k: int, threshold: float,
    ) -> tuple:
        """安全执行关键词检索（捕获异常）"""
        try:
            results = self.bm25.search(
                query=query,
                kb_ids=kb_ids,
                top_k=top_k,
                threshold=threshold,
            )
            return results, None
        except Exception as e:
            logger.error(f"Keyword search failed: {e}", exc_info=True)
            return [], str(e)
