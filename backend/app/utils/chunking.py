"""
自适应三层分块器 (Adaptive 3-Tier Chunking)

基于文档结构特征自动选择最优分块策略：
- Tier 1 (heading): 按 Markdown 标题边界分块，附加 breadcrumb 路径
- Tier 2 (heuristic): 按换页符/章节号/全大写标题分块（适合 PDF）
- Tier 3 (recursive): 递归分隔符分块（兜底策略）

支持 Parent-Child 分块模式：
- Child Chunk: 小块（默认 384 字符），用于向量检索精确匹配
- Parent Chunk: 大块（默认 4096 字符），返回给 LLM 获取完整上下文

参考文献：
- WeKnora docs/CHUNKING.md
- Vecta Feb-2026 基准测试：recursive splitting @512 tokens +15% overlap = 69% 准确性
"""

import re
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# 类型定义
# ═══════════════════════════════════════════════════════════

class ChunkStrategy(str, Enum):
    """分块策略枚举"""
    AUTO = "auto"            # 自动选择最佳策略
    HEADING = "heading"      # Tier 1: 按标题边界
    HEURISTIC = "heuristic"  # Tier 2: 按页面/章节标记
    RECURSIVE = "recursive"  # Tier 3: 递归分隔符
    LEGACY = "legacy"        # 兼容别名（等同于 recursive）


@dataclass
class ChunkingConfig:
    """分块配置"""
    chunk_size: int = 512              # 目标 Chunk 大小（字符数）
    chunk_overlap: int = 80            # Chunk 间重叠（字符数）
    separators: List[str] = field(default_factory=lambda: [
        "\n\n", "\n", "。", "！", "？", ";", "；", ". ", "! ", "? ",
    ])
    strategy: ChunkStrategy = ChunkStrategy.AUTO
    token_limit: int = 0               # Token 上限（0=不限制）
    languages: List[str] = field(default_factory=list)  # 文档语言提示

    # Parent-Child 配置
    enable_parent_child: bool = True
    parent_chunk_size: int = 4096      # 父块大小（字符）
    child_chunk_size: int = 384        # 子块大小（字符）


@dataclass
class Chunk:
    """分块结果"""
    content: str                       # 文本内容
    chunk_index: int                   # 序号
    chunk_type: str = "text"           # text / parent / child / faq
    parent_chunk_id: Optional[str] = None  # 父块 ID（子块用）
    breadcrumb: Optional[str] = None       # 章节路径（如 "# 概述 > ## 细节"）
    start_at: int = 0                  # 在原文中的起始位置
    end_at: int = 0                    # 在原文中的结束位置
    metadata: Dict = field(default_factory=dict)  # 扩展元信息


# ═══════════════════════════════════════════════════════════
# 文档画像分析器
# ═══════════════════════════════════════════════════════════

@dataclass
class DocumentProfile:
    """文档结构画像（用于自动策略选择）"""
    heading_count: int = 0         # Markdown 标题数
    form_feed_count: int = 0       # 换页符数
    chapter_marker_count: int = 0  # 章节标记数（中/英/德文）
    caps_title_count: int = 0      # 全大写标题数
    separator_count: int = 0       # 空行数
    total_lines: int = 0
    detected_languages: List[str] = field(default_factory=list)


class DocumentProfiler:
    """文档画像分析器 — 扫描文档结构特征"""

    # 多语言章节标记正则
    CHAPTER_PATTERNS = [
        r'(?:Chapter|CHAPTER|Kapitel|KAPITEL)\s+\d+',
        r'(?:第\s*[一二三四五六七八九十\d]+\s*[章节篇])',
    ]

    # 全大写标题行正则（至少 3 个单词，每词 ≥ 2 字符）
    CAPS_TITLE_RE = re.compile(
        r'^[A-Z][A-Z\s\-]{10,}$', re.MULTILINE
    )

    def profile(self, text: str) -> DocumentProfile:
        """分析文档结构特征"""
        p = DocumentProfile()
        lines = text.split('\n')
        p.total_lines = len(lines)

        for line in lines:
            stripped = line.strip()
            if not stripped:
                p.separator_count += 1
                continue

            # 检测 Markdown 标题
            if re.match(r'^#{1,6}\s', stripped):
                p.heading_count += 1

            # 检测换页符
            if '\f' in line:
                p.form_feed_count += 1

            # 检测章节标记
            for pattern in self.CHAPTER_PATTERNS:
                if re.search(pattern, stripped):
                    p.chapter_marker_count += 1

            # 检测全大写标题
            if self.CAPS_TITLE_RE.match(stripped):
                p.caps_title_count += 1

        return p


# ═══════════════════════════════════════════════════════════
# Tier 1: 标题感知分块器
# ═══════════════════════════════════════════════════════════

class HeadingChunker:
    """
    按 Markdown 标题边界分块

    策略：
    - 在 `#` / `##` / `###` 边界处断开
    - 每个 Chunk 前置面包屑路径（如 `# 产品 > ## API > ### 认证`）
    - 如果单个 section 过大，回退到递归分隔符分块

    面包屑在 Embedding 时被纳入，帮助语义检索定位到正确的文档章节。
    """

    HEADING_RE = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)

    def chunk(self, text: str, config: ChunkingConfig) -> List[Chunk]:
        """对 Markdown 文档进行标题感知分块"""
        sections = self._split_by_headings(text)
        chunks = []
        chunk_index = 0

        for section_title, section_content, heading_level in sections:
            breadcrumb = section_title or ""

            # 如果 section 内容小于当前配置的目标大小，直接作为一个 Chunk
            if len(section_content) <= config.chunk_size:
                chunks.append(Chunk(
                    content=self._prepend_breadcrumb(section_content, breadcrumb),
                    chunk_index=chunk_index,
                    breadcrumb=breadcrumb,
                    start_at=0,  # 简化：实际应计算位置
                    end_at=len(section_content),
                ))
                chunk_index += 1
            else:
                # Section 太大：用递归分隔符细切
                sub_chunks = self._recursive_split(
                    section_content, config, breadcrumb, chunk_index
                )
                chunks.extend(sub_chunks)
                chunk_index += len(sub_chunks)

        return chunks

    def _split_by_headings(
        self, text: str
    ) -> List[Tuple[str, str, int]]:
        """
        按标题拆分文档
        返回: [(标题, 内容, 层级), ...]
        """
        sections = []
        matches = list(self.HEADING_RE.finditer(text))
        if not matches:
            # 没有标题：整个文档当作一个 section
            sections.append(("", text, 0))
            return sections

        # 第一个标题之前的内容（如有）
        if matches[0].start() > 0:
            preamble = text[:matches[0].start()].strip()
            if preamble:
                sections.append(("", preamble, 0))

        for i, match in enumerate(matches):
            level = len(match.group(1))  # 标题层级（# 数量）
            title = match.group(2).strip()

            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            content = text[start:end].strip()

            if content:
                sections.append((title, content, level))

        return sections

    def _recursive_split(
        self, text: str, config: ChunkingConfig,
        breadcrumb: str, start_index: int
    ) -> List[Chunk]:
        """回退到递归分隔符分块"""
        recursive = RecursiveChunker()
        chunks = recursive.chunk(
            text,
            ChunkingConfig(
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
                separators=config.separators,
                strategy=ChunkStrategy.RECURSIVE,
                enable_parent_child=False,
            ),
        )
        # 附加 breadcrumb
        for i, chunk in enumerate(chunks):
            chunk.chunk_index = start_index + i
            chunk.breadcrumb = breadcrumb
            chunk.content = self._prepend_breadcrumb(chunk.content, breadcrumb)
        return chunks

    @staticmethod
    def _prepend_breadcrumb(content: str, breadcrumb: str) -> str:
        """在内容前添加面包屑路径"""
        if not breadcrumb:
            return content
        return f"[{breadcrumb}]\n{content}"


# ═══════════════════════════════════════════════════════════
# Tier 2: 启发式分块器
# ═══════════════════════════════════════════════════════════

class HeuristicChunker:
    """
    针对 PDF / 扫描文档的启发式分块器

    在以下位置断开：
    - 换页符（\f）
    - 编号章节（第一章、Chapter 1、1.、1.1.）
    - 全大写标题行
    - 空行聚集区
    """

    NUMBERED_SECTION_RE = re.compile(
        r'^(\d+(?:\.\d+)*)\s+[A-Z一-鿿]', re.MULTILINE
    )

    def chunk(self, text: str, config: ChunkingConfig) -> List[Chunk]:
        """启发式分块"""
        # Step 1: 在明确的结构边界处断开
        segments = self._split_at_boundaries(text)

        # Step 2: 对过大的段回退到递归分块
        chunks = []
        chunk_index = 0
        for segment, boundary_type in segments:
            if len(segment) <= config.chunk_size:
                chunks.append(Chunk(
                    content=segment,
                    chunk_index=chunk_index,
                    metadata={"boundary_type": boundary_type},
                ))
                chunk_index += 1
            else:
                recursive = RecursiveChunker()
                sub = recursive.chunk(
                    segment,
                    ChunkingConfig(
                        chunk_size=config.chunk_size,
                        chunk_overlap=config.chunk_overlap,
                        separators=config.separators,
                        strategy=ChunkStrategy.RECURSIVE,
                        enable_parent_child=False,
                    ),
                )
                for c in sub:
                    c.chunk_index = chunk_index
                    c.metadata["boundary_type"] = boundary_type
                    chunks.append(c)
                    chunk_index += 1

        return chunks

    def _split_at_boundaries(self, text: str) -> List[Tuple[str, str]]:
        """在结构边界处断开文档"""
        # 用一个综合正则匹配所有边界类型
        boundary_patterns = [
            (r'\f', 'form_feed'),                                    # 换页
            (r'\n(?=[A-Z][A-Z\s\-]{15,}\n)', 'caps_title'),         # 全大写标题
            (r'\n(?=\d+(?:\.\d+)*\s+[A-Z一-鿿])', 'numbered'),  # 编号章节
        ]

        combined_re = '|'.join(f'({p})' for p, _ in boundary_patterns)
        combined_re = '(' + combined_re + ')'

        segments = []
        last_end = 0
        for match in re.finditer(combined_re, text, re.MULTILINE):
            if match.start() > last_end:
                segment = text[last_end:match.start()].strip()
                if segment:
                    segments.append((segment, 'text'))

            last_end = match.end()

        # 最后一段
        if last_end < len(text):
            segment = text[last_end:].strip()
            if segment:
                segments.append((segment, 'text'))

        return segments


# ═══════════════════════════════════════════════════════════
# Tier 3: 递归分隔符分块器
# ═══════════════════════════════════════════════════════════

class RecursiveChunker:
    """
    递归分隔符分块器（兜底策略）

    按照优先级从高到低尝试分隔符：
    1. \\n\\n（段落边界）
    2. \\n（行边界）
    3. 。！？； （中文标点）
    4. .!?;    （英文标点）
    5. 空格     （实在找不到边界时）
    6. 字符级    （完全无分隔符时的暴力截断）

    这一策略来自 LangChain RecursiveCharacterTextSplitter 的核心思想，
    但在中文标点处理上进行了增强。
    """

    def chunk(self, text: str, config: ChunkingConfig) -> List[Chunk]:
        """递归分块"""
        chunks = []
        self._recursive_split(
            text, config.separators, 0,
            config.chunk_size, config.chunk_overlap, chunks,
        )

        # 编号
        for i, chunk in enumerate(chunks):
            chunk.chunk_index = i

        return chunks

    def _recursive_split(
        self,
        text: str,
        separators: List[str],
        depth: int,
        chunk_size: int,
        chunk_overlap: int,
        result: List[Chunk],
    ):
        """递归地使用分隔符切分文本"""
        if len(text) <= chunk_size:
            if text.strip():
                result.append(Chunk(content=text, chunk_index=0))
            return

        # 选择当前深度对应的分隔符
        separator = separators[depth] if depth < len(separators) else ""

        if separator:
            splits = text.split(separator)
        else:
            # 没有更多分隔符：按字符截断
            splits = [text]

        # 合并小块，切分大块
        current_chunk = ""
        for split in splits:
            if not split:
                continue

            # 当前块 + 新片段 是否超过大小限制？
            if len(current_chunk) + len(separator) + len(split) <= chunk_size:
                if current_chunk:
                    current_chunk += separator
                current_chunk += split
            else:
                # 当前块已满 → 保存
                if current_chunk.strip():
                    result.append(Chunk(content=current_chunk, chunk_index=0))

                # 新片段本身超过限制 → 递归细分
                if len(split) > chunk_size:
                    self._recursive_split(
                        split, separators, depth + 1,
                        chunk_size, chunk_overlap, result,
                    )
                    current_chunk = ""
                else:
                    current_chunk = split

        # 不丢失最后一个块
        if current_chunk.strip():
            result.append(Chunk(content=current_chunk, chunk_index=0))


# ═══════════════════════════════════════════════════════════
# 自适应分块编排器
# ═══════════════════════════════════════════════════════════

class AdaptiveChunker:
    """
    自适应分块编排器

    流程：
    1. 分析文档结构特征（DocumentProfiler）
    2. 选择最优策略（Auto 模式）
    3. 执行分块
    4. 可选：生成 Parent-Child 分块
    """

    # 策略选择阈值
    HEADING_MIN_THRESHOLD = 3     # 至少 3 个标题才用 heading 策略
    HEURISTIC_MIN_THRESHOLD = 1   # 至少 1 个结构信号才用 heuristic

    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig()
        self.profiler = DocumentProfiler()
        self.heading_chunker = HeadingChunker()
        self.heuristic_chunker = HeuristicChunker()
        self.recursive_chunker = RecursiveChunker()

    def chunk(self, text: str) -> List[Chunk]:
        """执行自适应分块"""
        if not text or not text.strip():
            return []

        strategy = self.config.strategy

        # Auto 模式：分析文档后自动选择
        if strategy == ChunkStrategy.AUTO:
            strategy = self._select_strategy(text)
            logger.info(f"Auto strategy selected: {strategy.value}")

        # 执行对应策略
        if strategy == ChunkStrategy.HEADING:
            chunks = self.heading_chunker.chunk(text, self.config)
        elif strategy == ChunkStrategy.HEURISTIC:
            chunks = self.heuristic_chunker.chunk(text, self.config)
        else:  # RECURSIVE / LEGACY
            chunks = self.recursive_chunker.chunk(text, self.config)

        # Parent-Child 分块
        if self.config.enable_parent_child and len(chunks) > 1:
            chunks = self._build_parent_child_chunks(text, chunks)

        return chunks

    def _select_strategy(self, text: str) -> ChunkStrategy:
        """根据文档画像自动选择分块策略"""
        profile = self.profiler.profile(text)

        logger.debug(
            f"Document profile: headings={profile.heading_count}, "
            f"formfeeds={profile.form_feed_count}, "
            f"chapters={profile.chapter_marker_count}, "
            f"caps_titles={profile.caps_title_count}, "
            f"lines={profile.total_lines}"
        )

        # 优先使用 heading 策略（Markdown 文档）
        if profile.heading_count >= self.HEADING_MIN_THRESHOLD:
            return ChunkStrategy.HEADING

        # 其次 heuristic 策略（PDF 文档）
        heuristic_signals = (
            profile.form_feed_count +
            profile.chapter_marker_count +
            profile.caps_title_count
        )
        if heuristic_signals >= self.HEURISTIC_MIN_THRESHOLD:
            return ChunkStrategy.HEURISTIC

        # 兜底递归策略
        return ChunkStrategy.RECURSIVE

    def _build_parent_child_chunks(
        self, text: str, child_chunks: List[Chunk]
    ) -> List[Chunk]:
        """
        构建 Parent-Child 分块结构

        Parent Chunk: 大块文本（parent_chunk_size），用于返回给 LLM
        Child Chunk: 小块文本（child_chunk_size），用于向量检索

        每个 Child Chunk 通过 parent_chunk_id 关联到其 Parent。
        """
        if self.config.child_chunk_size >= self.config.parent_chunk_size:
            logger.warning("child_chunk_size >= parent_chunk_size, skipping parent-child")
            return child_chunks

        import uuid

        # 生成 Parent Chunks（对原文按 parent_chunk_size 切分）
        parent_config = ChunkingConfig(
            chunk_size=self.config.parent_chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=self.config.separators,
            strategy=ChunkStrategy.RECURSIVE,
            enable_parent_child=False,
        )
        parent_chunks = self.recursive_chunker.chunk(text, parent_config)

        # 为每个 Parent 分配 ID
        for i, pc in enumerate(parent_chunks):
            pc.chunk_type = "parent"
            pc.chunk_index = i
            pc.metadata["parent_id"] = f"parent_{i}"

        # 为 Child Chunks 设置 parent_chunk_id
        # 简化处理：根据字符位置范围将子块映射到最近的父块
        parent_id_map = {}
        for pc in parent_chunks:
            parent_id = str(uuid.uuid4())
            pc.metadata["parent_uuid"] = parent_id
            parent_id_map[pc.chunk_index] = parent_id

        all_chunks = parent_chunks.copy()
        for i, cc in enumerate(child_chunks):
            cc.chunk_type = "child"
            cc.chunk_index = len(parent_chunks) + i
            # 简单策略：按比例分配子块到父块
            parent_idx = min(
                int(i / len(child_chunks) * len(parent_chunks)),
                len(parent_chunks) - 1,
            )
            cc.parent_chunk_id = parent_id_map.get(
                parent_chunks[parent_idx].chunk_index
            )
            all_chunks.append(cc)

        return all_chunks


# ═══════════════════════════════════════════════════════════
# 便捷函数
# ═══════════════════════════════════════════════════════════

def chunk_document(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 80,
    strategy: str = "auto",
    enable_parent_child: bool = True,
    parent_chunk_size: int = 4096,
    child_chunk_size: int = 384,
) -> List[Chunk]:
    """
    便捷函数：对文档进行分块

    Args:
        text: 文档文本内容
        chunk_size: 目标 Chunk 大小（字符数）
        chunk_overlap: Chunk 间重叠（字符数）
        strategy: 分块策略 (auto / heading / heuristic / recursive)
        enable_parent_child: 是否启用 Parent-Child 分块
        parent_chunk_size: 父块大小
        child_chunk_size: 子块大小

    Returns:
        分块列表
    """
    config = ChunkingConfig(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        strategy=ChunkStrategy(strategy),
        enable_parent_child=enable_parent_child,
        parent_chunk_size=parent_chunk_size,
        child_chunk_size=child_chunk_size,
    )
    chunker = AdaptiveChunker(config)
    return chunker.chunk(text)
