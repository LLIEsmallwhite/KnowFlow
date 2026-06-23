"""
KnowFlow 检索模块

提供完整的混合检索链路：
1. Dense Vector 检索 (Milvus)
2. BM25 关键词检索 (jieba + rank-bm25)
3. 动态 RRF 融合 (四因子自适应权重)
4. 多级去重 (ID/签名/重叠)
5. Cross-Encoder Rerank
6. 上下文合并
"""

from app.retrieval.milvus_client import MilvusClient, SearchResult
from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.bm25_retriever import BM25Retriever, BM25Result
from app.retrieval.hybrid_search import HybridSearchOrchestrator, HybridSearchResult
from app.retrieval.dynamic_rrf import DynamicRRF, DynamicWeightCalculator, RRFChunk, RRFWeights
from app.retrieval.dedup import MultiLevelDeduplicator, DedupStats

# Shared singleton BM25 — used by both upload (knowledge_base.py) and search (rag_pipeline.py)
shared_bm25 = BM25Retriever()

__all__ = [
    "MilvusClient",
    "SearchResult",
    "DenseRetriever",
    "BM25Retriever",
    "BM25Result",
    "HybridSearchOrchestrator",
    "HybridSearchResult",
    "DynamicRRF",
    "DynamicWeightCalculator",
    "RRFChunk",
    "RRFWeights",
    "MultiLevelDeduplicator",
    "DedupStats",
    "shared_bm25",
]
