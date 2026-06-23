"""
Search visualization component.

Displays retrieval stats: vector hits, keyword hits, RRF weights.
"""

import streamlit as st
from typing import Dict


def render_search_info(search_info: Dict):
    """Render search statistics below the chat message."""
    if not search_info:
        return

    v_hits = search_info.get("vector_hits", 0)
    k_hits = search_info.get("keyword_hits", 0)
    weights = search_info.get("rrf_weights", {})

    col1, col2, col3 = st.columns(3)
    col1.metric("向量命中", v_hits)
    col2.metric("关键词命中", k_hits)
    if weights:
        vw = weights.get("vector", 0)
        kw = weights.get("keyword", 0)
        col3.markdown(
            f"**RRF 权重**<br>"
            f"<span style='color:#2563eb'>向量 {vw:.2f}</span> / "
            f"<span style='color:#7c3aed'>关键词 {kw:.2f}</span>",
            unsafe_allow_html=True,
        )


def render_knowledge_refs(refs: list):
    """Render knowledge references as expandable cards."""
    if not refs:
        return

    with st.expander(f"📖 知识引用 ({len(refs)} 条)", expanded=False):
        for i, ref in enumerate(refs[:10]):
            st.markdown(
                f"""<div class="ref-card">
                <strong>[{i + 1}]</strong> 分数: {ref.get('score', 0):.4f}<br>
                <small>{ref.get('content_preview', '')[:300]}</small>
                </div>""",
                unsafe_allow_html=True,
            )
