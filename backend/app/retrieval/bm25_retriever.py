"""
BM25 关键词检索器

使用 jieba 中文分词 + rank-bm25 算法实现关键词检索。
BM25 对精确匹配（代码、ID、专有名词）特别有效，与 Dense 检索互补。

工作原理：
1. 对知识库所有 Chunk 预建 BM25 索引（内存中，按知识库分别建）
2. 查询时对查询词进行分词
3. 计算每个 Chunk 的 BM25 分数
4. 返回 Top-K 结果

性能考虑：
- BM25 索引存储在内存中，需在 Chunk 更新时重建
- 对于大知识库（>10万 Chunk），建议使用 Elasticsearch 替代内存方案
"""

import logging
import threading
from typing import List, Dict, Optional
from dataclasses import dataclass

import jieba
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


@dataclass
class BM25Result:
    """BM25 search result with document metadata."""
    chunk_id: str
    content: str
    score: float
    kb_id: str = ""
    doc_id: str = ""
    doc_title: str = ""
    doc_filename: str = ""


class BM25Retriever:
    """
    BM25 关键词检索器

    每个知识库维护独立的 BM25 索引。
    线程安全的索引管理，使用读写锁保护索引更新。

    使用示例:
        retriever = BM25Retriever()
        retriever.build_index("kb_001", chunks)  # chunks = [{"chunk_id": ..., "content": ...}]
        results = retriever.search("Kubernetes Pod", kb_ids=["kb_001"])
    """

    def __init__(self):
        # 每个知识库的索引数据
        self._indexes: Dict[str, BM25Okapi] = {}       # kb_id → BM25 索引
        self._chunks: Dict[str, List[Dict]] = {}       # kb_id → 原始 Chunk 列表
        self._lock = threading.RLock()                  # 读写锁

    # ─── 索引构建 ───
    def build_index(
        self,
        kb_id: str,
        chunks: List[Dict],
        force_rebuild: bool = False,
    ):
        """
        为知识库构建 BM25 索引

        Args:
            kb_id: 知识库 ID
            chunks: Chunk 列表 [{"chunk_id": id, "content": text, "doc_id": doc_id}, ...]
            force_rebuild: 是否强制重建（增量模式暂不支持，默认全量重建）

        索引构建时间：约 O(N * avg_doc_len)，N 为 Chunk 数
        """
        if not chunks:
            logger.warning(f"No chunks to index for KB '{kb_id}'")
            return

        with self._lock:
            # 对每个 Chunk 进行 jieba 分词
            tokenized = []
            for chunk in chunks:
                tokens = list(jieba.cut(chunk["content"]))
                tokenized.append(tokens)

            # 构建 BM25 索引
            bm25 = BM25Okapi(tokenized)

            self._indexes[kb_id] = bm25
            self._chunks[kb_id] = chunks

            logger.info(
                f"BM25 index built for KB '{kb_id}': "
                f"{len(chunks)} chunks, "
                f"avg tokens/chunk: ~{sum(len(t) for t in tokenized) // len(tokenized) if tokenized else 0}"
            )

    def remove_index(self, kb_id: str):
        """删除知识库的 BM25 索引"""
        with self._lock:
            self._indexes.pop(kb_id, None)
            self._chunks.pop(kb_id, None)
            logger.info(f"BM25 index removed for KB '{kb_id}'")

    # ─── 检索 ───
    def search(
        self,
        query: str,
        kb_ids: Optional[List[str]] = None,
        top_k: int = 50,
        threshold: float = 0.30,
    ) -> List[BM25Result]:
        """
        BM25 关键词检索

        Args:
            query: 查询文本
            kb_ids: 知识库 ID 列表（None = 搜索所有已索引的知识库）
            top_k: 返回数量上限
            threshold: 最低分数阈值

        Returns:
            检索结果列表（按 BM25 分数降序）
        """
        # tokenize 查询
        tokenized_query = list(jieba.cut(query))

        if not tokenized_query:
            return []

        with self._lock:
            # Determine KBs to search (empty list = search all)
            all_indexed = list(self._indexes.keys())
            if not kb_ids:
                kb_ids = all_indexed
            else:
                kb_ids = [kb for kb in kb_ids if kb in self._indexes]

            if not kb_ids:
                logger.warning(
                    "No BM25 indexes for search. kb_ids=%s, indexed=%s",
                    kb_ids, all_indexed,
                )
                return []

            # 对每个知识库执行检索
            all_scores: List[tuple] = []  # (chunk, score, kb_id)

            for kb_id in kb_ids:
                bm25 = self._indexes[kb_id]
                chunks = self._chunks[kb_id]
                scores = bm25.get_scores(tokenized_query)

                for i, score in enumerate(scores):
                    if score >= threshold:
                        all_scores.append((chunks[i], float(score), kb_id))

            # 按分数降序排列
            all_scores.sort(key=lambda x: x[1], reverse=True)

            # 截取 Top-K
            results = []
            for chunk, score, kb_id in all_scores[:top_k]:
                results.append(BM25Result(
                    chunk_id=chunk.get("chunk_id", ""),
                    content=chunk.get("content", ""),
                    score=score,
                    kb_id=kb_id,
                    doc_id=chunk.get("doc_id", ""),
                    doc_title=chunk.get("doc_title", ""),
                    doc_filename=chunk.get("doc_filename", ""),
                ))

            logger.debug(
                f"BM25 search: query='{query[:50]}...', "
                f"kbs={kb_ids}, results={len(results)}"
            )
            return results

    # ─── 索引状态查询 ───
    def get_indexed_kbs(self) -> List[str]:
        """获取已建索引的知识库列表"""
        with self._lock:
            return list(self._indexes.keys())

    def get_index_size(self, kb_id: str) -> int:
        """获取知识库的索引大小"""
        with self._lock:
            return len(self._chunks.get(kb_id, []))

    def is_indexed(self, kb_id: str) -> bool:
        """检查知识库是否已建索引"""
        with self._lock:
            return kb_id in self._indexes


# ─── 批量建索引入口 ───
def build_bm25_index_from_db(
    kb_id: str,
    chunks: List[Dict],
    retriever: Optional[BM25Retriever] = None,
) -> BM25Retriever:
    """
    便捷函数：从数据库 Chunk 数据构建 BM25 索引

    Args:
        kb_id: 知识库 ID
        chunks: Chunk 记录列表（SQLAlchemy 对象或字典）
        retriever: 已有检索器（None 则创建新实例）

    Returns:
        BM25Retriever 实例
    """
    if retriever is None:
        retriever = BM25Retriever()

    # Convert to BM25Retriever expected format with doc metadata
    formatted = [
        {
            "chunk_id": c.get("id", c.get("chunk_id", "")),
            "content": c.get("content", ""),
            "doc_id": c.get("document_id", c.get("doc_id", "")),
            "doc_title": c.get("doc_title", ""),
            "doc_filename": c.get("doc_filename", ""),
        }
        for c in chunks
    ]

    retriever.build_index(kb_id, formatted)
    return retriever
