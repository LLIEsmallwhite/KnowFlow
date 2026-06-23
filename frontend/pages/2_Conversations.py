"""
Conversation History Page

View, load, and manage past conversation sessions.
"""

import streamlit as st
from utils.session import init_session_state, get_api

init_session_state()

st.set_page_config(page_title="对话历史", page_icon="💾", layout="wide")

st.title("💾 对话历史")
st.caption("查看和管理历史对话会话")

api = get_api()

if st.button("🔄 刷新", use_container_width=False):
    st.rerun()

try:
    sessions = api.list_sessions()
except Exception as e:
    st.error(f"无法加载会话: {e}")
    sessions = []

if not sessions:
    st.info("暂无对话记录")
else:
    for s in sessions:
        with st.container(border=True):
            c1, c2, c3, c4 = st.columns([4, 2, 1, 1])
            with c1:
                st.markdown(f"**{s['title'][:60]}**")
                st.caption(f"创建: {s.get('created_at', '')[:19]}")
            with c2:
                st.caption(f"💬 {s.get('message_count', 0)} 条消息 · "
                           f"🪙 {s.get('total_tokens', 0)} tokens")
            with c3:
                if st.button("📂 加载", key=f"load_{s['id']}"):
                    # Load session into main chat state
                    st.session_state.current_session_id = s["id"]
                    try:
                        msgs = api.get_messages(s["id"])
                        st.session_state.messages = [
                            {"role": m["role"], "content": m["content"]}
                            for m in msgs
                        ]
                        st.success("已加载对话，切换到主页查看")
                        st.switch_page("app.py")
                    except Exception as e:
                        st.error(f"加载失败: {e}")
            with c4:
                if st.button("🗑️", key=f"del_s_{s['id']}"):
                    try:
                        api.delete_session(s["id"])
                        st.success("已删除")
                        st.rerun()
                    except Exception as e:
                        st.error(f"删除失败: {e}")

# ─── Stats ───
st.markdown("---")
st.markdown("### 📊 统计")
col_a, col_b, col_c = st.columns(3)
col_a.metric("总会话数", len(sessions))
col_b.metric("总消息数", sum(s.get("message_count", 0) for s in sessions))
col_c.metric("总 Token", sum(s.get("total_tokens", 0) for s in sessions))
