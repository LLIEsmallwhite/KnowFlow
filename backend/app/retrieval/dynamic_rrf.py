"""
动态权重 RRF（Reciprocal Rank Fusion）融合算法

本项目核心创新点 ⭐：
  区别于业界固定权重（Vector:Keyword = 0.7:0.3），
  基于四因子自适应调整融合权重：
    1. 查询类型（精确查询 → 提升关键词；概念查询 → 提升向量）
    2. 查询长度（短查询语义信号弱 → 微调）
    3. 结果分布（某路检索极少 → 自动补偿另一路）
    4. 分数方差（方差大 = 区分度高 → 加权）

RRF 公式:
  RRF_score(chunk) = VectorWeight / (k + VectorRank) + KeywordWeight / (k + KeywordRank)

  其中 k = 60（平滑常数，使第 1 名和第 100 名的分值不会差异过大）

为什么选 RRF 而非加权求和？
  1. BM25 分数（0-∞）和 Cosine Similarity（0-1）量纲不可比
  2. RRF 只依赖排名，天然消除量纲差异
  3. 零训练成本，与 Elasticsearch 8.x / Weaviate 方案一致

参考资料:
  - Cormack et al., "Reciprocal Rank Fusion outperforms Condorcet..." (SIGIR 2009)
  - Elasticsearch 8.x Hybrid Search documentation
  - WeKnora internal/application/service/knowledgebase_search_fusion.go
"""

import re
import math
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

from app.retrieval.milvus_client import SearchResult as DenseResult
from app.retrieval.bm25_retriever import BM25Result
from app.core.config import settings

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# 数据类型
# ═══════════════════════════════════════════════════════════

@dataclass
class RRFChunk:
    """RRF 融合后的单个 Chunk"""
    chunk_id: str
    content: str
    rrf_score: float
    vector_rank: int = -1           # 在向量检索中的排名（1-indexed，-1 表示未命中）
    keyword_rank: int = -1          # 在关键词检索中的排名
    source: str = ""                 # "vector" / "keyword" / "both"
    kb_id: str = ""
    doc_id: str = ""
    # 保留原始分数用于调试
    original_vector_score: float = 0.0
    original_keyword_score: float = 0.0

    @property
    def is_in_both(self) -> bool:
        """是否同时被两路检索命中"""
        return self.vector_rank > 0 and self.keyword_rank > 0


@dataclass
class RRFWeights:
    """RRF 动态权重"""
    vector: float
    keyword: float

    def __repr__(self):
        return f"RRFWeights(vector={self.vector:.3f}, keyword={self.keyword:.3f})"


# ═══════════════════════════════════════════════════════════
# 动态权重计算
# ═══════════════════════════════════════════════════════════

class DynamicWeightCalculator:
    """
    动态权重计算器

    四因子模型：
    1. 查询类型分类（准确/概念/混合）
    2. 查询长度补偿
    3. 结果分布补偿
    4. 分数方差评估
    """

    # 权重范围限制
    MIN_WEIGHT = 0.15
    MAX_WEIGHT = 0.85

    def __init__(self):
        # 基础权重
        self.base_vector_weight = settings.RRF_VECTOR_WEIGHT_BASE
        self.base_keyword_weight = settings.RRF_KEYWORD_WEIGHT_BASE

    def compute(
        self,
        query: str,
        vector_results: List[DenseResult],
        keyword_results: List[BM25Result],
    ) -> RRFWeights:
        """
        计算动态 RRF 权重

        Args:
            query: 查询文本
            vector_results: 向量检索结果
            keyword_results: 关键词检索结果

        Returns:
            RRFWeights（且 vector + keyword = 1.0）
        """
        vec_w = self.base_vector_weight
        kw_w = self.base_keyword_weight

        # ── 因子 1：查询类型分类 ──
        query_type = self._classify_query(query)
        adjustments = []

        if query_type == "exact":
            kw_w += 0.10
            vec_w -= 0.10
            adjustments.append("exact_query: +0.10 keyword")
        elif query_type == "conceptual":
            vec_w += 0.08
            kw_w -= 0.08
            adjustments.append("conceptual_query: +0.08 vector")

        # ── 因子 2：查询长度 ──
        word_count = len(query.split())
        if word_count <= 5:
            kw_w += 0.05
            vec_w -= 0.05
            adjustments.append(f"short_query({word_count}w): +0.05 keyword")
        elif word_count >= 20:
            vec_w += 0.05
            kw_w -= 0.05
            adjustments.append(f"long_query({word_count}w): +0.05 vector")

        # ── 因子 3：结果分布补偿 ──
        n_vec = len(vector_results)
        n_kw = len(keyword_results)

        if n_vec <= 3 and n_kw > 10:
            kw_w += 0.10
            vec_w -= 0.10
            adjustments.append(f"sparse_vector({n_vec}hits): +0.10 keyword")
        elif n_kw <= 3 and n_vec > 10:
            vec_w += 0.10
            kw_w -= 0.10
            adjustments.append(f"sparse_keyword({n_kw}hits): +0.10 vector")

        # ── 因子 4：分数方差 ──
        if n_vec >= 5:
            vec_scores = [r.score for r in vector_results[:10]]
            vec_variance = float(np.var(vec_scores))
            if vec_variance > 0.05:
                vec_w += 0.05
                kw_w -= 0.05
                adjustments.append(f"high_vector_variance({vec_variance:.3f}): +0.05 vector")

        if n_kw >= 5:
            kw_scores = [r.score for r in keyword_results[:10]]
            kw_variance = float(np.var(kw_scores))
            if kw_variance > 0.05:
                kw_w += 0.05
                vec_w -= 0.05
                adjustments.append(f"high_keyword_variance({kw_variance:.3f}): +0.05 keyword")

        # ── 权重裁剪 ──
        vec_w = max(self.MIN_WEIGHT, min(self.MAX_WEIGHT, vec_w))
        kw_w = max(self.MIN_WEIGHT, min(self.MAX_WEIGHT, kw_w))

        # ── 归一化到 1.0 ──
        total = vec_w + kw_w
        vec_w /= total
        kw_w /= total

        weights = RRFWeights(vector=round(vec_w, 4), keyword=round(kw_w, 4))

        # 日志
        logger.info(
            f"Dynamic RRF weights: {weights} | "
            f"query='{query[:60]}...' type={query_type} | "
            f"vector_hits={n_vec} keyword_hits={n_kw}"
        )
        if adjustments:
            logger.debug(f"Adjustments: {'; '.join(adjustments)}")

        return weights

    def _classify_query(self, query: str) -> str:
        """
        基于规则的查询分类

        精确查询特征:
        - 全大写缩写 (API, SDK, HTTP)
        - 编号/JIRA 号 (JIRA-1234)
        - 长数字 ID
        - 版本号 (v2.3)
        - 引号包裹的精确短语

        概念查询特征:
        - 疑问词（如何/为什么/怎样/什么是）
        - 描述性请求（总结/概述/比较/区别/原理）
        """
        # 精确匹配特征
        exact_patterns = [
            (r'\b[A-Z]{2,8}\b', 'acronym'),           # 全大写缩写
            (r'\b[A-Z]+-\d+\b', 'issue_id'),           # 编号
            (r'\b\d{6,}\b', 'long_number'),            # 长数字
            (r'\bv\d+\.\d+(\.\d+)?', 'version'),       # 版本号
            (r'"(.*?)"', 'quoted_phrase'),             # 精确短语
        ]
        exact_count = sum(
            1 for p, _ in exact_patterns if re.search(p, query)
        )

        # 概念查询特征
        conceptual_keywords = [
            '什么是', '如何', '为什么', '怎样', '怎么',
            '区别', '比较', '总结', '概述', '原理',
            '流程', '步骤', '方法', '方案', '最佳实践',
            'what is', 'how to', 'why', 'explain',
            'describe', 'compare', 'summary', 'overview',
        ]
        conceptual_count = sum(
            1 for kw in conceptual_keywords if kw.lower() in query.lower()
        )

        if exact_count >= 2:
            return "exact"
        elif conceptual_count >= 1:
            return "conceptual"
        else:
            return "mixed"


# ═══════════════════════════════════════════════════════════
# RRF 融合器
# ═══════════════════════════════════════════════════════════

class DynamicRRF:
    """
    动态权重 RRF 融合器

    工作流程：
    1. 计算动态权重（DynamicWeightCalculator）
    2. 构建排名映射（按各自分数排序，1-indexed）
    3. 汇总所有唯一 Chunk
    4. 计算加权 RRF 分数
    5. 按 RRF 分数降序返回

    使用示例:
        rrf = DynamicRRF(k=60)
        results = rrf.fuse(
            query="Kubernetes Pod 无法启动",
            vector_results=vector_hits,
            keyword_results=keyword_hits,
        )
    """

    def __init__(self, k: int = None):
        """
        Args:
            k: RRF 平滑常数（默认从配置读取 60）
        """
        self.k = k or settings.RRF_K
        self.weight_calc = DynamicWeightCalculator()

    def fuse(
        self,
        query: str,
        vector_results: List[DenseResult],
        keyword_results: List[BM25Result],
    ) -> List[RRFChunk]:
        """
        执行动态 RRF 融合

        注意：
        - 如果只有单路结果，直接返回去重后的结果（保持原始分数语义）
        - 如果两路都有结果，执行加权 RRF 融合

        Args:
            query: 查询文本（用于动态权重计算）
            vector_results: 向量检索结果（已按分数降序排列）
            keyword_results: 关键词检索结果（已按分数降序排列）

        Returns:
            RRF 融合后的 Chunk 列表（按 RRF 分数降序）
        """
        # 单路结果：直接返回
        if not vector_results and not keyword_results:
            return []

        if not keyword_results:
            logger.info("Only vector results available — returning as-is")
            return self._single_source_results(vector_results, "vector")

        if not vector_results:
            logger.info("Only keyword results available — returning as-is")
            return self._single_source_results(keyword_results, "keyword")

        # 计算动态权重
        weights = self.weight_calc.compute(query, vector_results, keyword_results)

        # 构建排名映射（按原始分数 1-indexed 排名）
        vector_ranks: Dict[str, int] = {}
        for i, r in enumerate(vector_results):
            if r.chunk_id not in vector_ranks:
                vector_ranks[r.chunk_id] = i + 1

        keyword_ranks: Dict[str, int] = {}
        for i, r in enumerate(keyword_results):
            if r.chunk_id not in keyword_ranks:
                keyword_ranks[r.chunk_id] = i + 1

        # 汇总所有唯一 Chunk
        chunk_map: Dict[str, RRFChunk] = {}
        for r in vector_results:
            chunk_map[r.chunk_id] = RRFChunk(
                chunk_id=r.chunk_id,
                content=r.content,
                rrf_score=0.0,
                vector_rank=vector_ranks.get(r.chunk_id, -1),
                keyword_rank=-1,
                source="vector",
                kb_id=r.kb_id,
                doc_id=r.doc_id,
                original_vector_score=r.score,
            )
        for r in keyword_results:
            if r.chunk_id in chunk_map:
                chunk_map[r.chunk_id].keyword_rank = keyword_ranks.get(r.chunk_id, -1)
                chunk_map[r.chunk_id].source = "both"
                chunk_map[r.chunk_id].original_keyword_score = r.score
            else:
                chunk_map[r.chunk_id] = RRFChunk(
                    chunk_id=r.chunk_id,
                    content=r.content,
                    rrf_score=0.0,
                    vector_rank=-1,
                    keyword_rank=keyword_ranks.get(r.chunk_id, -1),
                    source="keyword",
                    kb_id=r.kb_id,
                    doc_id=r.doc_id,
                    original_keyword_score=r.score,
                )

        # 计算加权 RRF 分数
        for chunk_id, chunk in chunk_map.items():
            rrf_score = 0.0
            if chunk.vector_rank > 0:
                rrf_score += weights.vector / (self.k + chunk.vector_rank)
            if chunk.keyword_rank > 0:
                rrf_score += weights.keyword / (self.k + chunk.keyword_rank)
            chunk.rrf_score = round(rrf_score, 6)

        # 按 RRF 分数降序排列
        results = sorted(
            chunk_map.values(),
            key=lambda x: x.rrf_score,
            reverse=True,
        )

        # 日志
        both_count = sum(1 for r in results if r.is_in_both)
        logger.info(
            f"RRF fusion complete: {len(results)} unique chunks "
            f"({both_count} in both, "
            f"{sum(1 for r in results if r.source=='vector')} vector-only, "
            f"{sum(1 for r in results if r.source=='keyword')} keyword-only) | "
            f"weights={weights}"
        )

        # 打印 Top-5 用于调试
        for i, chunk in enumerate(results[:5]):
            logger.debug(
                f"  RRf #{i+1}: chunk={chunk.chunk_id[:8]}... "
                f"score={chunk.rrf_score:.6f} "
                f"v_rank={chunk.vector_rank} k_rank={chunk.keyword_rank} "
                f"src={chunk.source}"
            )

        return results

    def _single_source_results(
        self, results: list, source: str,
    ) -> List[RRFChunk]:
        """
        单路检索结果 → RRFChunk 格式

        不计算 RRF 分数，保留原始分数语义。
        """
        chunks = []
        for i, r in enumerate(results):
            chunk = RRFChunk(
                chunk_id=r.chunk_id,
                content=r.content,
                rrf_score=r.score,  # 保留原始分数而非 RRF
                vector_rank=i + 1 if source == "vector" else -1,
                keyword_rank=i + 1 if source == "keyword" else -1,
                source=source,
                kb_id=r.kb_id if hasattr(r, 'kb_id') else "",
                doc_id=r.doc_id if hasattr(r, 'doc_id') else "",
                original_vector_score=r.score if source == "vector" else 0.0,
                original_keyword_score=r.score if source == "keyword" else 0.0,
            )
            chunks.append(chunk)
        return chunks
