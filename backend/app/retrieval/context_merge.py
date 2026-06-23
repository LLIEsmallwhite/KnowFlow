"""
上下文合并模块

将 Rerank 后的 Chunk 列表合并为 LLM 可用的最终上下文。

功能：
1. Chunk 排序和分组（按知识来源分组）
2. 相邻 Chunk 合并（连续的 ChunkIndex 合并为更完整段落）
3. 上下文格式化（生成带引用标记的 Prompt 模板文本）
4. 短上下文扩展（过短的 Chunk 追加邻近内容）

设计参考:
    - WeKnora chat_pipeline/merge.go
    - WeKnora chat_pipeline/merge_expand.go
"""

import logging
from typing import List, Dict, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class MergedContext:
    """合并后的上下文块"""
    content: str                      # 文本内容
    chunk_ids: List[str] = field(default_factory=list)   # 引用的 Chunk ID 列表
    knowledge_title: str = ""         # 来源文档标题
    relevance_score: float = 0.0      # 综合相关性分数
    source_type: str = "document"     # document / faq / web

    @property
    def reference_marker(self) -> str:
        """生成引用标记文本"""
        ids_short = [cid[:8] for cid in self.chunk_ids[:3]]
        return f"[Ref: {', '.join(ids_short)}]"


class ContextMerger:
    """
    上下文合并器

    将 Rerank 后的结果组织为 LLM 易于理解的结构化上下文。

    处理流程：
    1. 按知识来源分组（同一篇文档的 Chunk 聚集）
    2. 组内按 ChunkIndex 排序
    3. 相邻 Chunk 合并（减少碎片化）
    4. 按相关性得分插入到最终上下文
    """

    # 相邻 Chunk 合并的最大间距（ChunkIndex 差距 ≤ 此值视为相邻）
    ADJACENT_GAP = 2

    # 单个 Chunk 展示的最短长度（字符）
    MIN_CHUNK_LENGTH = 350

    # 单个 Chunk 展示的最大长度（字符，超出截断）
    MAX_CHUNK_LENGTH = 850

    def merge(
        self,
        reranked_results: List,
        top_k: int = 10,
    ) -> List[MergedContext]:
        """
        合并 Rerank 后的结果为结构化上下文

        Args:
            reranked_results: Rerank 后的结果列表（需有 content, chunk_id, score 等属性）
            top_k: 最终保留的上下文块数量

        Returns:
            MergedContext 列表
        """
        if not reranked_results:
            return []

        # Step 1: 按知识来源分组
        groups = self._group_by_source(reranked_results)

        # Step 2: 合并组内相邻 Chunk
        merged = []
        for source_key, chunks in groups.items():
            group_merged = self._merge_adjacent(chunks)
            merged.extend(group_merged)

        # Step 3: 按相关性分数重新排序
        merged.sort(key=lambda x: x.relevance_score, reverse=True)

        # Step 4: 截取 Top-K
        merged = merged[:top_k]

        logger.info(
            f"Context merge: {len(reranked_results)} chunks → "
            f"{len(merged)} context blocks (top_k={top_k})"
        )

        return merged

    def _group_by_source(self, results: List) -> Dict[str, List]:
        """
        按知识来源分组

        分组键: (knowledge_id 或 doc_id)
        这确保同一篇文档的 Chunk 在上下文中相邻展示。
        """
        groups: Dict[str, List] = {}
        for r in results:
            source_id = getattr(r, 'doc_id', '') or getattr(r, 'kb_id', '') or 'unknown'
            if source_id not in groups:
                groups[source_id] = []
            groups[source_id].append(r)
        return groups

    def _merge_adjacent(self, chunks: List) -> List[MergedContext]:
        """
        合并组内相邻的 Chunk

        相邻判定：ChunkIndex 差距 ≤ ADJACENT_GAP
        """
        if not chunks:
            return []

        # 按 ChunkIndex 排序（如果有的话）
        sorted_chunks = sorted(
            chunks,
            key=lambda x: getattr(x, 'chunk_index', 0) or 0,
        )

        merged = []
        current_group = [sorted_chunks[0]]
        current_score = _get_score(sorted_chunks[0])

        for chunk in sorted_chunks[1:]:
            prev_idx = getattr(current_group[-1], 'chunk_index', 0) or 0
            curr_idx = getattr(chunk, 'chunk_index', 0) or 0
            score = _get_score(chunk)

            # 判断是否与上一组相邻
            if curr_idx - prev_idx <= self.ADJACENT_GAP:
                current_group.append(chunk)
                current_score = max(current_score, score)
            else:
                # 保存当前组并开始新组
                merged.append(self._build_context_block(current_group, current_score))
                current_group = [chunk]
                current_score = score

        # 不丢失最后一组
        if current_group:
            merged.append(self._build_context_block(current_group, current_score))

        return merged

    def _build_context_block(
        self, chunks: List, score: float,
    ) -> MergedContext:
        """
        从 Chunk 组构建 MergedContext

        合并策略：
        - 所有 Chunk 内容去重拼接（用分隔符隔开）
        - 保留所有 Chunk ID 用于引用追踪
        """
        # 去重拼接内容
        seen_content = set()
        unique_contents = []
        for c in chunks:
            content = getattr(c, 'content', '')
            content_key = content[:100]  # 前 100 字符作为简易去重键
            if content_key not in seen_content:
                seen_content.add(content_key)
                unique_contents.append(content)

        merged_content = "\n\n---\n\n".join(unique_contents)

        # 收集所有 Chunk ID
        chunk_ids = [getattr(c, 'chunk_id', '') for c in chunks]

        # 获取来源标题
        knowledge_title = ""
        for c in chunks:
            title = getattr(c, 'knowledge_title', '') or getattr(c, 'doc_title', '')
            if title:
                knowledge_title = title
                break

        return MergedContext(
            content=merged_content,
            chunk_ids=chunk_ids,
            knowledge_title=knowledge_title,
            relevance_score=score,
        )

    def format_for_llm(
        self,
        contexts: List[MergedContext],
        max_total_length: int = 8000,
    ) -> str:
        """
        格式化上下文为 LLM Prompt 模板文本

        格式：
        [Reference 1] (来源: xxx, 分数: 0.95)
        文本内容...

        [Reference 2] (来源: yyy, 分数: 0.87)
        文本内容...

        Args:
            contexts: MergedContext 列表
            max_total_length: 总文本长度上限（超出时自动截断）

        Returns:
            格式化后的上下文字符串
        """
        if not contexts:
            return "（未找到相关文档）"

        parts = []
        total_len = 0

        for i, ctx in enumerate(contexts, 1):
            source = ctx.knowledge_title or "未知来源"
            score = ctx.relevance_score

            header = f"[Reference {i}] (来源: {source}, 相关性: {score:.3f})\n"
            body = ctx.content

            block = header + body
            block_len = len(block)

            # 检查是否超出总长度限制
            if total_len + block_len > max_total_length:
                remaining = max_total_length - total_len
                if remaining > 200:
                    block = block[:remaining] + "\n... (truncated)"
                else:
                    break

            parts.append(block)
            total_len += len(block)

        return "\n\n".join(parts)


def _get_score(obj) -> float:
    """兼容多种类型的分数提取"""
    return (
        getattr(obj, 'rerank_score', 0.0)
        or getattr(obj, 'rrf_score', 0.0)
        or getattr(obj, 'score', 0.0)
    )
