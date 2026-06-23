"""
文本处理工具集

提供文本归一化、内容签名生成、Token 重叠计算等基础功能。
这些功能在多级去重和文本预处理中共享使用。
"""

import re
import hashlib
from typing import List


# ─── 文本归一化 ───

def normalize_content(text: str) -> str:
    """
    文本归一化处理

    处理步骤：
    1. 转小写
    2. 合并连续空白字符为单个空格
    3. 移除标点符号（保留中文字符和字母数字）
    4. 去除首尾空白

    Args:
        text: 原始文本

    Returns:
        归一化后的文本
    """
    if not text:
        return ""

    text = text.lower().strip()
    # 合并所有空白字符为单个空格
    text = re.sub(r'\s+', ' ', text)
    # 移除标点（保留中文字符 一-鿿、字母、数字、空格）
    text = re.sub(r'[^\w\s一-鿿　-〿＀-￯]', '', text)
    # 再次合并可能因标点移除产生的多余空白
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ─── 内容签名 ───

def build_content_signature(content: str) -> str:
    """
    构建内容签名（归一化后 SHA256）

    用于快速判断两个 Chunk 是否为近似重复内容。
    取 SHA256 的前 16 位十六进制字符作为签名。

    Args:
        content: 原始文本内容

    Returns:
        16 位十六进制签名（空字符串表示输入无效）
    """
    normalized = normalize_content(content)
    if not normalized or len(normalized) < 10:
        return ""
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


# ─── Token 重叠计算 ───

def content_overlap_ratio(text_a: str, text_b: str) -> float:
    """
    计算两段文本的 Token 重叠系数

    使用简单的空格分 Token 方式（对中英文混合场景适用）。
    重叠系数 = |tokens(a) ∩ tokens(b)| / min(|tokens(a)|, |tokens(b)|)

    Args:
        text_a: 文本 A
        text_b: 文本 B

    Returns:
        重叠比例 [0.0, 1.0]
    """
    tokens_a = set(text_a.split())
    tokens_b = set(text_b.split())

    if not tokens_a or not tokens_b:
        return 0.0

    smaller_size = min(len(tokens_a), len(tokens_b))
    intersection = len(tokens_a & tokens_b)

    return intersection / smaller_size if smaller_size > 0 else 0.0


def is_content_contained(short_text: str, long_text: str) -> bool:
    """
    判断短文本是否为长文本的子串（归一化后）

    用于去重检测：如果短文本的全部内容都在长文本中出现，
    则短文本是冗余的。

    Args:
        short_text: 较短的文本
        long_text: 较长的文本

    Returns:
        True 如果短文本是长文本的子串
    """
    short_norm = normalize_content(short_text)
    long_norm = normalize_content(long_text)

    if len(short_norm) < 10 or len(long_norm) < 20:
        return False

    return short_norm in long_norm


# ─── 安全截断 ───

def truncate_text(text: str, max_chars: int) -> str:
    """
    安全截断文本（按字符而非字节）

    用于 Embedding 输入截断、Prompt 压缩等场景。

    Args:
        text: 原始文本
        max_chars: 最大字符数

    Returns:
        截断后的文本（超出部分用 ... 标记）
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def sanitize_for_embedding(text: str, max_chars: int = 20000) -> str:
    """
    Embedding 输入清洗

    处理：
    1. 移除 Base64 图片载荷（避免无意义的字节串进入 Embedding）
    2. 按字符上限截断

    Args:
        text: 原始文本
        max_chars: 最大字符数（默认 20000）

    Returns:
        清洗并截断后的文本
    """
    # 移除 Base64 编码的图片数据
    # 正则匹配 data:image/...;base64,... 的 base64 载荷部分
    text = re.sub(
        r'data:image/[a-z0-9.+-]+;base64,[a-zA-Z0-9+/=]{200,}',
        '[image]',
        text,
    )

    # 移除 Markdown 图片语法中的 base64 数据
    text = re.sub(
        r'!\[([^\]]*)\]\(\s*data:image/[a-z0-9.+-]+;base64,[^)]+\)',
        r'[image: \1]',
        text,
    )

    # HTML img 标签中的 base64
    text = re.sub(
        r'<img\b[^>]*\bsrc=["\']\s*data:image/[a-z0-9.+-]+;base64,[^"\']+["\'][^>]*>',
        '[image]',
        text,
    )

    # 按字符截断
    if len(text) > max_chars:
        text = text[:max_chars]

    return text


# ─── 日志安全 ───

def sanitize_for_log(text: str, max_len: int = 200) -> str:
    """
    日志安全清洗

    移除控制字符（CR/LF/Tab），防止日志注入。
    截断过长的查询字符串。

    Args:
        text: 原始文本
        max_len: 最大长度

    Returns:
        安全的日志文本
    """
    if not text:
        return ""
    # 移除控制字符
    text = text.replace('\r', '\\r').replace('\n', '\\n').replace('\t', '\\t')
    if len(text) > max_len:
        text = text[:max_len] + "..."
    return text
