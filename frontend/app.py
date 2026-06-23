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

    # 模拟助手回复（后续步骤接入真实 API）
    with st.chat_message("assistant"):
        with st.spinner("🔍 正在检索知识库..."):
            import time
            time.sleep(0.5)  # 模拟延迟
        response = f"收到您的问题：「{query}」\n\n> ⚠️ 后端 API 尚未连接（将在后续步骤实现）\n\n当前项目骨架已搭建完成：\n- ✅ FastAPI 后端就绪\n- ✅ Streamlit 前端就绪\n- ⏳ Milvus 集成 (Step 4)\n- ⏳ 动态 RRF 融合 (Step 5)"
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
