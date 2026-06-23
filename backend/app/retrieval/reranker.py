"""
Cross-Encoder Reranker（重排序器）

在 RRF 融合后对候选 Chunk 进行精排：
- 使用 Cross-Encoder 模型直接对 (query, chunk) 对打分
- 比 Bi-Encoder（向量检索）更精确，但速度更慢
- 只在候选集较小（< 50）时使用

支持的模型：
- 本地: sentence-transformers (cross-encoder/ms-marco-MiniLM-L-6-v2)
- 云端: Cohere Rerank API (rerank-multilingual-v3.0)

设计参考:
    - WeKnora chat_pipeline/rerank.go
    - Nogueira et al., "Passage Re-ranking with BERT" (2019)
"""

import logging
from typing import List, Optional, Tuple
from dataclasses import dataclass

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class RerankResult:
    """重排序后的单个结果"""
    chunk_id: str
    content: str
    original_score: float        # RRF 融合后的分数
    rerank_score: float           # Cross-Encoder 重排序分数
    kb_id: str = ""
    doc_id: str = ""
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseReranker:
    """重排序器基类"""

    def rerank(
        self,
        query: str,
        candidates: List,
        top_k: int = 10,
        threshold: float = 0.2,
    ) -> List[RerankResult]:
        """
        对候选列表进行重排序

        Args:
            query: 查询文本
            candidates: 候选列表（RRFChunk / SearchResult）
            top_k: 返回数量上限
            threshold: 最低分数阈值

        Returns:
            重排序后的结果（按 rerank_score 降序，不超过 top_k 个）
        """
        raise NotImplementedError


class LocalCrossEncoderReranker(BaseReranker):
    """
    本地 Cross-Encoder 重排序器

    使用 sentence-transformers 的 CrossEncoder 模型。
    首次加载会下载模型（约 100MB），后续使用缓存。

    推荐模型:
    - ms-marco-MiniLM-L-6-v2: 英文为主，速度快
    - BAAI/bge-reranker-v2-m3: 多语言（中英文），精度高
    """

    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.RERANK_MODEL
        self._model = None
        logger.info(f"Reranker initialized with model: {self.model_name}")

    @property
    def model(self):
        """懒加载 CrossEncoder 模型"""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
                logger.info(f"CrossEncoder model loaded: {self.model_name}")
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Run: pip install sentence-transformers"
                )
            except Exception as e:
                logger.error(f"Failed to load reranker model: {e}")
                raise
        return self._model

    def rerank(
        self,
        query: str,
        candidates: List,
        top_k: int = 10,
        threshold: float = 0.2,
    ) -> List[RerankResult]:
        """
        Cross-Encoder 重排序

        Args:
            query: 查询文本
            candidates: 候选列表（需有 .content 或 .chunk_content 属性）
            top_k: 返回数量上限
            threshold: 最低重排序分数（范围取决于模型，通常 -10 到 10）

        Returns:
            按 rerank_score 降序的 Top-K 结果
        """
        if not candidates:
            return []

        # 提取候选文本
        candidate_texts = []
        for c in candidates:
            content = getattr(c, 'content', '') or getattr(c, 'chunk_content', '')
            candidate_texts.append(content)

        # 截断过长文本（Cross-Encoder 通常有 512 token 限制）
        max_len = 450  # 给 query 留空间
        truncated_texts = [
            text[:max_len] if len(text) > max_len else text
            for text in candidate_texts
        ]

        # 构建 (query, text) 对
        pairs = [(query, text) for text in truncated_texts]

        # 批量打分
        try:
            scores = self.model.predict(
                pairs,
                batch_size=16,
                show_progress_bar=False,
            )
        except Exception as e:
            logger.error(f"Rerank prediction failed: {e}")
            # 失败时返回原始排序
            return _fallback_rerank(candidates, top_k)

        # 组装结果
        results = []
        for candidate, score in zip(candidates, scores):
            score_val = float(score)
            if score_val >= threshold:
                chunk_id = getattr(candidate, 'chunk_id', '')
                kb_id = getattr(candidate, 'kb_id', '')
                doc_id = getattr(candidate, 'doc_id', '')
                original_score = getattr(candidate, 'rrf_score', 0.0) or getattr(candidate, 'score', 0.0)

                results.append(RerankResult(
                    chunk_id=chunk_id,
                    content=candidate_texts[candidates.index(candidate)],
                    original_score=original_score,
                    rerank_score=score_val,
                    kb_id=kb_id,
                    doc_id=doc_id,
                ))

        # 按重排序分数降序排列
        results.sort(key=lambda x: x.rerank_score, reverse=True)

        # 截取 Top-K
        before = len(results)
        results = results[:top_k]

        logger.info(
            f"Rerank: {len(candidates)} candidates → {before} above threshold → "
            f"{len(results)} returned (top_k={top_k}, threshold={threshold})"
        )
        return results


class NoOpReranker(BaseReranker):
    """
    空重排序器（不进行 Rerank，直接透传）

    用于以下场景：
    - 未配置 Rerank 模型
    - 候选数过少无需重排序
    - 开发调试阶段
    """

    def rerank(
        self,
        query: str,
        candidates: List,
        top_k: int = 10,
        threshold: float = 0.2,
    ) -> List[RerankResult]:
        """直接透传，不重排序"""
        results = []
        for c in candidates[:top_k]:
            chunk_id = getattr(c, 'chunk_id', '')
            content = getattr(c, 'content', '')
            score = getattr(c, 'rrf_score', 0.0) or getattr(c, 'score', 0.0)
            kb_id = getattr(c, 'kb_id', '')
            doc_id = getattr(c, 'doc_id', '')

            if score >= threshold:
                results.append(RerankResult(
                    chunk_id=chunk_id,
                    content=content,
                    original_score=score,
                    rerank_score=score,
                    kb_id=kb_id,
                    doc_id=doc_id,
                ))
        return results[:top_k]


def _fallback_rerank(candidates: List, top_k: int) -> List[RerankResult]:
    """Rerank 失败时的回退方案（保持原序）"""
    results = []
    for c in candidates[:top_k]:
        results.append(RerankResult(
            chunk_id=getattr(c, 'chunk_id', ''),
            content=getattr(c, 'content', ''),
            original_score=getattr(c, 'rrf_score', 0.0) or getattr(c, 'score', 0.0),
            rerank_score=getattr(c, 'rrf_score', 0.0) or getattr(c, 'score', 0.0),
            kb_id=getattr(c, 'kb_id', ''),
            doc_id=getattr(c, 'doc_id', ''),
        ))
    return results


def create_reranker(provider: str = None) -> BaseReranker:
    """
    工厂函数：根据配置创建重排序器

    Args:
        provider: 重排序模型类型 (local / cohere / none)

    Returns:
        重排序器实例
    """
    provider = provider or settings.RERANK_PROVIDER

    if provider == "local":
        return LocalCrossEncoderReranker()
    elif provider == "cohere":
        # TODO: 实现 Cohere Rerank API 集成
        logger.warning("Cohere Rerank not implemented yet, falling back to NoOp")
        return NoOpReranker()
    elif provider == "none":
        return NoOpReranker()
    else:
        logger.warning(f"Unknown rerank provider: {provider}, using NoOp")
        return NoOpReranker()
