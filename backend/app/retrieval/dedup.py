"""
多级去重模块

在 RRF 融合后与 Rerank 之前，对检索结果进行三级去重：

Level 1: Chunk ID 去重
    - 相同 ChunkID 只保留最高分的一个
    - 发生在 RRF 融合阶段（RRF 已按 chunk_id 聚合）

Level 2: 内容签名去重
    - 归一化文本 → SHA256 前 16 位 → 相同签名 = 近似重复
    - 捕获跨文档、跨 ChunkIndex 的重复内容

Level 3: Token 重叠去重
    - 计算两个 Chunk 的 Token 重叠系数
    - overlap ≥ 85% → 保留分数更高的
    - 捕获改写/格式变化导致的近似重复

设计参考:
    - WeKnora search.go removeDuplicateResults / removePartialOverlaps
    - SimHash / MinHash 思想的轻量级替代
"""

import re
import hashlib
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# 内容重叠阈值（Token 重叠系数 ≥ 此值视为重复）
OVERLAP_THRESHOLD = 0.85

# 内容签名最短长度（过短文本不计算签名）
MIN_SIGNATURE_LENGTH = 10


@dataclass
class DedupStats:
    """去重统计"""
    input_count: int = 0
    id_removed: int = 0
    signature_removed: int = 0
    overlap_removed: int = 0

    @property
    def output_count(self) -> int:
        return self.input_count - self.id_removed - self.signature_removed - self.overlap_removed

    @property
    def removal_rate(self) -> float:
        if self.input_count == 0:
            return 0.0
        return 1.0 - (self.output_count / self.input_count)


class MultiLevelDeduplicator:
    """
    三级去重器

    使用方式:
        dedup = MultiLevelDeduplicator()
        clean_results, stats = dedup.deduplicate(rrf_results)
    """

    def __init__(self, overlap_threshold: float = OVERLAP_THRESHOLD):
        self.overlap_threshold = overlap_threshold

    def deduplicate(self, results: List) -> tuple:
        """
        执行多级去重

        Args:
            results: RRFChunk / SearchResult 列表（需有 chunk_id, content, score 属性）

        Returns:
            (去重后的结果列表, DedupStats)
        """
        stats = DedupStats(input_count=len(results))

        if not results:
            return [], stats

        # Level 1: Chunk ID 去重
        results = self._dedup_by_id(results)
        stats.id_removed = stats.input_count - len(results)

        # Level 2: 内容签名去重
        before_sig = len(results)
        results = self._dedup_by_signature(results)
        stats.signature_removed = before_sig - len(results)

        # Level 3: Token 重叠去重
        before_overlap = len(results)
        results = self._dedup_by_overlap(results)
        stats.overlap_removed = before_overlap - len(results)

        logger.info(
            f"Dedup: {stats.input_count} → {stats.output_count} "
            f"(id={stats.id_removed}, sig={stats.signature_removed}, "
            f"overlap={stats.overlap_removed}, "
            f"rate={stats.removal_rate:.1%})"
        )

        return results, stats

    # ─── Level 1: ID 去重 ───
    @staticmethod
    def _dedup_by_id(results: List) -> List:
        """相同 Chunk ID 只保留最高分"""
        seen: Dict[str, any] = {}
        for r in results:
            cid = r.chunk_id
            if cid not in seen or _get_score(r) > _get_score(seen[cid]):
                seen[cid] = r
        # 保持原始排序
        seen_ids = set(r.chunk_id for r in results)
        deduped = [seen[cid] for cid in seen_ids if cid in seen]
        deduped.sort(key=_get_score, reverse=True)
        return deduped

    # ─── Level 2: 签名去重 ───
    @staticmethod
    def _dedup_by_signature(results: List) -> List:
        """内容签名去重"""
        seen_sigs: Dict[str, any] = {}
        for r in results:
            sig = _build_signature(r.content)
            if sig and sig not in seen_sigs:
                seen_sigs[sig] = r
        return list(seen_sigs.values())

    # ─── Level 3: 重叠去重 ───
    def _dedup_by_overlap(self, results: List) -> List:
        """
        Token 重叠去重

        对于 overlap ≥ threshold 的 Chunk 对：
        - 保留分数更高的
        - 如果分数相同，保留更长的
        """
        if len(results) <= 1:
            return results

        normalized = [self._normalize(r.content) for r in results]
        removed = set()
        kept_results = []

        for i in range(len(results)):
            if i in removed:
                continue

            for j in range(i + 1, len(results)):
                if j in removed:
                    continue

                overlap = _token_overlap(normalized[i], normalized[j])

                if overlap >= self.overlap_threshold:
                    # 决策：保留哪个？
                    score_i = _get_score(results[i])
                    score_j = _get_score(results[j])

                    if score_i >= score_j:
                        removed.add(j)
                        logger.debug(
                            f"Overlap dedup: kept {results[i].chunk_id[:8]}... "
                            f"(score={score_i:.4f}), dropped {results[j].chunk_id[:8]}... "
                            f"(score={score_j:.4f}), overlap={overlap:.2f}"
                        )
                    else:
                        removed.add(i)
                        logger.debug(
                            f"Overlap dedup: kept {results[j].chunk_id[:8]}... "
                            f"(score={score_j:.4f}), dropped {results[i].chunk_id[:8]}... "
                            f"(score={score_i:.4f}), overlap={overlap:.2f}"
                        )
                        break  # i 被移除，跳过剩余 j

            if i not in removed:
                kept_results.append(results[i])

        return kept_results

    @staticmethod
    def _normalize(text: str) -> str:
        """
        文本归一化（用于 Token 重叠计算）

        处理：
        1. 转小写
        2. 合并空白字符
        3. 移除标点符号（保留字母数字和中文）
        4. 去首尾空白
        """
        if not text:
            return ""
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s一-鿿]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()


# ═══════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════

def _get_score(obj) -> float:
    """从结果对象中提取分数（兼容多种类型）"""
    if hasattr(obj, 'rrf_score'):
        return obj.rrf_score
    elif hasattr(obj, 'score'):
        return obj.score
    return 0.0


def _build_signature(content: str) -> Optional[str]:
    """
    构建内容签名

    归一化 → SHA256 → 前 16 位十六进制
    """
    if not content or len(content.strip()) < MIN_SIGNATURE_LENGTH:
        return None

    # 归一化
    text = content.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s一-鿿]', '', text)

    if len(text) < MIN_SIGNATURE_LENGTH:
        return None

    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _token_overlap(text_a: str, text_b: str) -> float:
    """
    计算 Token 重叠系数

    使用简单空格分词（对英文有效，对中文需要 jieba 但此处权衡性能）。
    重叠系数 = |A ∩ B| / min(|A|, |B|)
    """
    tokens_a = set(text_a.split())
    tokens_b = set(text_b.split())

    if not tokens_a or not tokens_b:
        return 0.0

    smaller = min(len(tokens_a), len(tokens_b))
    intersection = len(tokens_a & tokens_b)

    return intersection / smaller if smaller > 0 else 0.0
