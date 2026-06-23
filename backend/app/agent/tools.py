"""
Agent 工具定义

提供 ReAct Agent 可调用的工具集：
- knowledge_search: 知识库检索
- web_search: 联网搜索
"""

import json
import logging
from typing import List, Dict, Any, Optional
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def knowledge_search(query: str, kb_ids: Optional[List[str]] = None) -> str:
    """
    在知识库中搜索相关文档。

    Args:
        query: 搜索查询文本
        kb_ids: 可选的知识库 ID 列表，不传则搜索所有知识库

    Returns:
        JSON 格式的搜索结果
    """
    try:
        from app.graph.rag_pipeline import invoke_rag_pipeline
        result = invoke_rag_pipeline(query=query, kb_ids=kb_ids or [])
        refs = result.get("knowledge_refs", [])
        if not refs:
            return json.dumps({"status": "no_results", "message": "未找到相关文档"}, ensure_ascii=False)
        return json.dumps({
            "status": "success",
            "hit_count": len(refs),
            "results": [
                {"content": r.get("content_preview", "")[:500], "score": r.get("score", 0)}
                for r in refs[:5]
            ],
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)[:300]}, ensure_ascii=False)


# ─── 所有可用工具列表 ───
ALL_TOOLS = [knowledge_search]
