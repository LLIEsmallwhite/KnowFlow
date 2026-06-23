"""
KnowFlow — Streamlit 前端入口

启动方式:
    streamlit run app.py
"""

import streamlit as st

# ─── 页面全局配置 — 必须放在最前面 ───
st.set_page_config(
    page_title="KnowFlow — RAG 知识库问答助手",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── 自定义 CSS 样式 ───
st.markdown("""
<style>
    /* 全局变量 */
    :root {
        --primary: #2563eb;
        --primary-dark: #1d4ed8;
        --bg-sidebar: #f8fafc;
    }

    /* 隐藏 Streamlit 默认元素 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header[data-testid="stHeader"] {background-color: transparent;}

    /* 主标题样式 */
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #2563eb, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }

    /* 知识引用卡片 */
    .ref-card {
        border-left: 3px solid #2563eb;
        background: #f0f4ff;
        padding: 12px 16px;
        margin: 6px 0;
        border-radius: 4px;
        font-size: 0.9rem;
    }

    /* 检索可视化容器 */
    .viz-container {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        margin: 12px 0;
    }

    /* RRF 分数标签 */
    .rrf-score {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .rrf-score.high { background: #dcfce7; color: #166534; }
    .rrf-score.medium { background: #fef3c7; color: #92400e; }
    .rrf-score.low { background: #fee2e2; color: #991b1b; }
</style>
""", unsafe_allow_html=True)

# ─── 侧边栏 ───
with st.sidebar:
    st.markdown("## 🤖 KnowFlow")
    st.markdown("---")

    # 知识库选择
    st.markdown("### 📚 知识库")
    kb_options = ["全部知识库", "产品文档", "技术规范", "运维手册"]
    selected_kb = st.selectbox("选择知识库", kb_options, label_visibility="collapsed")

    st.markdown("---")

    # 对话历史
    st.markdown("### 💾 对话历史")
    st.button("➕ 新建对话", use_container_width=True)

    # 示例历史条目
    with st.expander("📁 最近对话", expanded=True):
        st.markdown("- K8s 集群排查")
        st.markdown("- 数据库优化方案")
        st.markdown("- API 文档说明")

    st.markdown("---")

    # 设置
    st.markdown("### ⚙️ 设置")
    st.toggle("🌐 联网搜索", value=False)
    st.toggle("🧠 记忆模式", value=True)
    st.selectbox("模型", ["deepseek-chat", "deepseek-v4-pro", "qwen-plus"])

# ─── 主区域 ───
st.markdown('<p class="main-title">💬 智能问答</p>', unsafe_allow_html=True)
st.caption("基于动态 RRF 混合检索的企业级 RAG 知识库问答助手")

# ─── 对话展示区 ───
# 初始化消息历史
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "您好！我是 KnowFlow 知识库助手 🤖\n\n我可以帮您：\n- 🔍 检索企业内部文档\n- 📊 分析技术规范和流程\n- 💡 回答运维和开发问题\n\n请随时向我提问！"}
    ]

# 渲染聊天记录
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ─── 输入区域 ───
col_query, col_btn = st.columns([8, 1])
with col_query:
    query = st.chat_input("请输入您的问题...")
with col_btn:
    st.markdown("")  # 占位，保持对齐

if query:
    # 添加用户消息
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # 调用 KnowFlow RAG API
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        import httpx
        import os

        backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
        api_url = f"{backend_url}/api/v1/chat/stream"

        try:
            with st.spinner("🔍 正在检索知识库..."):
                with httpx.stream(
                    "POST",
                    api_url,
                    json={
                        "query": query,
                        "kb_ids": [],
                        "stream": True,
                        "top_k": 10,
                    },
                    timeout=60.0,
                ) as resp:
                    if resp.status_code != 200:
                        full_response = f"❌ API 错误 ({resp.status_code}): 请确认后端已启动"
                        response_placeholder.markdown(full_response)
                    else:
                        import json
                        search_info = {}
                        buffer = ""
                        for line in resp.iter_lines():
                            if line.startswith("data: "):
                                data_str = line[6:]
                                if data_str == "[DONE]":
                                    break
                                try:
                                    data = json.loads(data_str)
                                    msg_type = data.get("type", "")
                                    if msg_type == "token":
                                        buffer += data.get("data", "")
                                        response_placeholder.markdown(buffer + "▌")
                                    elif msg_type == "search_info":
                                        search_info = data.get("data", {})
                                except json.JSONDecodeError:
                                    pass

                        full_response = buffer or "（未收到回答）"

                        # 展示检索信息
                        if search_info:
                            v_hits = search_info.get("vector_hits", 0)
                            k_hits = search_info.get("keyword_hits", 0)
                            weights = search_info.get("rrf_weights", {})
                            caption = f"🔍 向量命中 {v_hits} · 关键词命中 {k_hits} · RRF权重 {weights.get('vector', 0):.2f}/{weights.get('keyword', 0):.2f}"
                            st.caption(caption)

        except httpx.ConnectError:
            full_response = "❌ 无法连接后端服务。请确认 `uvicorn main:app --reload` 已在 http://localhost:8000 启动。"
            response_placeholder.markdown(full_response)
        except Exception as e:
            full_response = f"❌ 请求出错: {str(e)[:200]}"
            response_placeholder.markdown(full_response)

        if full_response:
            response_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
