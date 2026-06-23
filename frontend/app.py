"""
KnowFlow — Enterprise RAG Knowledge Base Q&A Assistant
Streamlit Frontend Entry Point

Start: streamlit run app.py
"""

import streamlit as st

# ─── Page config (MUST be first) ───
st.set_page_config(
    page_title="💬 智能问答",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)



# ─── Custom CSS ───
st.markdown("""
<style>
    :root {
        --primary: #2563eb;
        --primary-dark: #1d4ed8;
        --bg-sidebar: #f8fafc;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header[data-testid="stHeader"] {background-color: transparent;}

    .main-title {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #2563eb, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .ref-card {
        border-left: 3px solid #2563eb;
        background: #f0f4ff;
        padding: 12px 16px;
        margin: 6px 0;
        border-radius: 4px;
        font-size: 0.9rem;
    }
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

# ─── Init session state ───
from utils.session import init_session_state, get_api, start_new_chat
init_session_state()

# ─── Sidebar ───
from components.sidebar import render_sidebar
render_sidebar()

# ─── Main Area ───
st.markdown('<p class="main-title">💬 智能问答</p>', unsafe_allow_html=True)
st.caption("基于动态 RRF 混合检索的企业级 RAG 知识库问答助手")

# Init default welcome message
if not st.session_state.messages:
    st.session_state.messages = [
        {"role": "assistant",
         "content": "您好！我是 KnowFlow 知识库助手 🤖\n\n我可以帮您：\n- 🔍 检索已上传的文档内容\n- 📊 分析技术规范和流程\n- 💡 回答运维和开发问题\n\n请先在 **知识库管理页** 上传文档，然后开始提问！"}
    ]

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ─── Chat Input ───
query = st.chat_input("请输入您的问题...")

if query:
    api = get_api()
    sid = st.session_state.current_session_id

    # Add user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Stream response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        search_info = {}
        knowledge_refs = []

        try:
            with st.spinner("🔍 正在检索知识库..."):
                for event in api.chat_stream(
                    query=query,
                    session_id=sid,
                    kb_ids=st.session_state.selected_kb_ids,
                    enable_web_search=st.session_state.enable_web_search,
                    enable_memory=st.session_state.enable_memory,
                    temperature=st.session_state.temperature,
                ):
                    etype = event.get("type", "")
                    if etype == "token":
                        full_response += event.get("data", "")
                        response_placeholder.markdown(full_response + "▌")
                    elif etype == "search_info":
                        search_info = event.get("data", {})
                    elif etype == "knowledge_refs":
                        knowledge_refs = event.get("data", [])
                    elif etype == "done":
                        done_data = event.get("data", {})
                        if not full_response:
                            full_response = done_data.get("full_answer", "")
                        if not search_info:
                            search_info = done_data.get("search_info", {})
                    elif etype == "warning":
                        st.warning(event.get("data", ""))
                    elif etype == "error":
                        st.error(event.get("data", "未知错误"))

                # Update session_id from first response
                if not st.session_state.current_session_id:
                    st.session_state.current_session_id = sid

        except Exception as e:
            full_response = f"❌ 连接后端失败: {str(e)[:200]}\n\n请确认后端已启动: `uvicorn main:app --reload`"

        if not full_response:
            full_response = "（未收到回答，请重试）"

        response_placeholder.markdown(full_response)

        # Show search info
        if search_info:
            from components.search_viz import render_search_info
            render_search_info(search_info)

        # Show knowledge references grouped by document
        if knowledge_refs:
            from components.search_viz import render_knowledge_refs
            render_knowledge_refs(knowledge_refs)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
