"""
Search visualization component with document references and hover previews.
"""

import streamlit as st
from typing import Dict, List
from collections import defaultdict


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
            f"**RRF** "
            f"<span style='color:#2563eb'>V {vw:.2f}</span> / "
            f"<span style='color:#7c3aed'>K {kw:.2f}</span>",
            unsafe_allow_html=True,
        )


def render_knowledge_refs(refs: List[Dict]):
    """Render knowledge references grouped by document with hover previews."""
    if not refs:
        return

    # Group refs by document
    groups = defaultdict(list)
    for r in refs:
        key = r.get("doc_filename") or r.get("doc_title") or "未知文档"
        groups[key].append(r)

    total_refs = len(refs)
    doc_count = len(groups)

    with st.expander(
        f"📖 知识引用 — {total_refs} 条片段，来自 {doc_count} 个文档", expanded=False
    ):
        for doc_name, doc_refs in groups.items():
            st.markdown(f"**📄 {doc_name}** ({len(doc_refs)} 条)")
            for i, ref in enumerate(doc_refs[:5]):
                content = ref.get("content_preview", "")[:300]
                full = ref.get("full_content", content)
                score = ref.get("score", 0)

                st.markdown(
                    f"""<div class="ref-card">
                    <strong>#{i + 1}</strong> 相关度: {score:.4f}
                    <details><summary style="cursor:pointer">{content[:120]}...</summary>
                    <p style="margin-top:8px; padding:8px; background:#f0f4ff; border-radius:4px; font-size:0.85rem; white-space:pre-wrap;">{full}</p>
                    </details>
                    </div>""",
                    unsafe_allow_html=True,
                )
            if len(doc_refs) > 5:
                st.caption(f"... 还有 {len(doc_refs) - 5} 条")
