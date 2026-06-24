"""
Knowledge Base Management Page

Create, list, delete KBs. Upload and manage documents.
"""

import os
import tempfile
import streamlit as st
from utils.session import init_session_state, get_api, load_kb_list

init_session_state()

st.set_page_config(page_title="知识库管理", page_icon="📚", layout="wide")

st.title("📚 知识库管理")
st.caption("创建知识库、上传文档、管理索引")

api = get_api()

# ─── Tabs ───
tab_kb, tab_upload = st.tabs(["📁 知识库列表", "📤 文档上传"])

# ─── Tab 1: KB List ───
with tab_kb:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("知识库")
    with col2:
        if st.button("🔄 刷新列表", use_container_width=True):
            load_kb_list()
            st.rerun()

    # Create KB form
    with st.expander("➕ 创建知识库", expanded=False):
        with st.form("create_kb_form"):
            c1, c2 = st.columns(2)
            with c1:
                kb_name = st.text_input("名称", placeholder="例如：产品文档")
                kb_type = st.selectbox("类型", ["document", "faq", "wiki"])
            with c2:
                kb_dept = st.selectbox("部门", ["_通用", "engineering", "product", "hr", "finance", "ops"])
                kb_sec = st.selectbox("密级", ["0-公开", "1-内部", "2-机密", "3-绝密"], index=1)
            kb_desc = st.text_area("描述（可选）", placeholder="知识库用途说明")
            submitted = st.form_submit_button("创建", use_container_width=True)
            if submitted and kb_name:
                try:
                    dept = kb_dept.split("-")[0] if "-" in kb_dept else "_"
                    sec = int(kb_sec.split("-")[0])
                    result = api.create_kb(name=kb_name, description=kb_desc or None,
                                           kb_type=kb_type, department=dept, security_level=sec)
                    st.success(f"✅ 知识库 '{result['name']}' 创建成功")
                    load_kb_list()
                    st.rerun()
                except Exception as e:
                    st.error(f"创建失败: {e}")

    # List KBs
    kbs = api.list_kbs()
    if not kbs:
        st.info("暂无知识库，点击上方按钮创建")
    else:
        for kb in kbs:
            with st.container(border=True):
                c1, c2, c3 = st.columns([4, 2, 1])
                with c1:
                    st.markdown(f"**{kb['name']}**")
                    if kb.get("description"):
                        st.caption(kb["description"])
                with c2:
                    st.caption(f"📄 {kb.get('document_count', 0)} 文档 | "
                               f"🧩 {kb.get('chunk_count', 0)} 片段 | "
                               f"📦 {kb.get('kb_type', 'document')}")
                with c3:
                    if st.button("🗑️", key=f"del_{kb['id']}",
                                 help="删除知识库"):
                        try:
                            api.delete_kb(kb["id"])
                            st.success("已删除")
                            load_kb_list()
                            st.rerun()
                        except Exception as e:
                            st.error(f"删除失败: {e}")

                # Documents list
                try:
                    docs = api.list_documents(kb["id"])
                    if docs:
                        with st.expander(f"📄 文档列表 ({len(docs)})"):
                            for doc in docs:
                                dc1, dc2, dc3 = st.columns([4, 2, 1])
                                with dc1:
                                    status_icon = {"completed": "✅", "processing": "⏳",
                                                   "failed": "❌", "pending": "⬜"}.get(
                                        doc.get("status", ""), "❓")
                                    st.markdown(f"{status_icon} {doc['title']}")
                                with dc2:
                                    st.caption(f"{doc.get('file_type', '')} · "
                                               f"{doc.get('chunk_count', 0)} chunks")
                                with dc3:
                                    if st.button("🗑️", key=f"del_doc_{doc['id']}"):
                                        try:
                                            api.delete_document(kb["id"], doc["id"])
                                            st.success("文档已删除")
                                            st.rerun()
                                        except Exception as e:
                                            st.error(f"删除失败: {e}")
                    else:
                        st.caption("暂无文档")
                except Exception:
                    st.caption("无法加载文档列表")


# ─── Tab 2: Upload ───
with tab_upload:
    st.subheader("📤 上传文档")

    # Reload KB list
    kbs = api.list_kbs()
    if not kbs:
        st.warning("请先创建知识库再上传文档")
    else:
        kb_options = {kb["name"]: kb["id"] for kb in kbs}
        selected_kb_name = st.selectbox("目标知识库", list(kb_options.keys()))
        target_kb_id = kb_options[selected_kb_name]

        uploaded_files = st.file_uploader(
            "选择文件",
            type=["pdf", "docx", "doc", "md", "markdown", "html", "htm",
                  "txt", "text", "csv", "xlsx", "xls", "pptx", "ppt",
                  "png", "jpg", "jpeg"],
            accept_multiple_files=True,
            help="支持 PDF、Word、Markdown、HTML、TXT、CSV、Excel、PPT、图片",
        )

        if uploaded_files:
            for uf in uploaded_files:
                with st.status(f"处理中: {uf.name}...", expanded=True) as status:
                    try:
                        # Save to temp file
                        suffix = os.path.splitext(uf.name)[1] or ".txt"
                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix=suffix
                        ) as tmp:
                            tmp.write(uf.read())
                            tmp_path = tmp.name
                        uf.seek(0)

                        result = api.upload_document(
                            target_kb_id,
                            tmp_path,
                            title=uf.name,
                            original_name=uf.name,
                        )

                        try:
                            os.unlink(tmp_path)
                        except Exception:
                            pass

                        status.update(label=f"✅ {uf.name} — 已上传", state="complete")
                    except Exception as e:
                        status.update(label=f"❌ {uf.name} 失败: {str(e)[:150]}", state="error")

            load_kb_list()
            st.success("所有文件处理完成！可切换到会话页开始提问")
