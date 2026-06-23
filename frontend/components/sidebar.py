"""
Sidebar component for KnowFlow.

Renders the left sidebar with:
- KB selector (loaded from API)
- Session list
- Settings panel
- New chat button
"""

import streamlit as st
from utils.session import get_api, load_kb_list, start_new_chat


def render_sidebar():
    """Render the main app sidebar. Returns selected settings."""
    api = get_api()

    with st.sidebar:
        st.markdown("## 🤖 KnowFlow")
        st.markdown("---")

        # ─── Knowledge Base Selector ───
        st.markdown("### 📚 知识库")

        # Refresh KB list
        if st.button("🔄 刷新", use_container_width=True, key="refresh_kbs"):
            load_kb_list()
            st.rerun()

        # Lazy-load KB list
        if not st.session_state.kbs_loaded:
            load_kb_list()

        kb_list = st.session_state.sidebar_kb_list
        if kb_list:
            kb_names = ["全部知识库"] + [kb["name"] for kb in kb_list]
            kb_ids_map = {"全部知识库": []}
            for kb in kb_list:
                kb_ids_map[kb["name"]] = [kb["id"]]

            selected_name = st.selectbox(
                "选择知识库",
                kb_names,
                label_visibility="collapsed",
            )
            st.session_state.selected_kb_ids = kb_ids_map.get(selected_name, [])
        else:
            st.caption("暂无知识库，请在管理页创建")
            st.session_state.selected_kb_ids = []

        st.markdown("---")

        # ─── Session Management ───
        st.markdown("### 💾 对话")
        if st.button("➕ 新建对话", use_container_width=True):
            start_new_chat()
            st.rerun()

        # List recent sessions
        try:
            sessions = api.list_sessions()
            if sessions:
                session_options = {s["title"][:40]: s["id"] for s in sessions[:10]}
                selected_title = st.selectbox(
                    "历史对话",
                    ["（新对话）"] + list(session_options.keys()),
                    label_visibility="collapsed",
                )
                if selected_title != "（新对话）":
                    sid = session_options[selected_title]
                    if sid != st.session_state.current_session_id:
                        st.session_state.current_session_id = sid
                        # Load messages from API
                        try:
                            msgs = api.get_messages(sid)
                            st.session_state.messages = [
                                {"role": m["role"], "content": m["content"]}
                                for m in msgs
                            ]
                            st.rerun()
                        except Exception:
                            pass
        except Exception:
            st.caption("无法加载对话列表")

        st.markdown("---")

        # ─── Settings ───
        st.markdown("### ⚙️ 设置")

        # Web search disabled (blocked in China)
        # web_search = st.toggle("联网搜索", value=st.session_state.enable_web_search)

        memory_mode = st.toggle(
            "🧠 记忆模式", value=st.session_state.enable_memory,
        )
        st.session_state.enable_memory = memory_mode

        model = st.selectbox(
            "模型",
            ["deepseek-chat", "deepseek-v4-pro", "qwen-plus"],
            index=0 if st.session_state.selected_model == "deepseek-chat" else
                  1 if st.session_state.selected_model == "deepseek-v4-pro" else 2,
        )
        st.session_state.selected_model = model

        st.markdown("---")
        st.caption(f"后端: {api.base_url}")
