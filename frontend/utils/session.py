"""
Session State Manager for Streamlit

Initializes and manages all session state variables across pages.
Call init_session_state() at the top of app.py.
"""

import streamlit as st
from utils.api import KnowFlowAPI


def init_session_state():
    """Initialize all session state defaults if not already set."""
    defaults = {
        "api": None,
        "messages": [],
        "current_session_id": None,
        "selected_kb_ids": [],
        "enable_web_search": False,
        "enable_memory": True,
        "selected_model": "deepseek-chat",
        "temperature": 0.1,
        "sidebar_kb_list": [],
        "kbs_loaded": False,
        # Auth
        "logged_in": False,
        "auth_token": None,
        "username": "",
    }

    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # Init API client with stored token
    if st.session_state.api is None:
        st.session_state.api = KnowFlowAPI(token=st.session_state.auth_token)


def get_api() -> KnowFlowAPI:
    """Get the global API client."""
    return st.session_state.api


def reset_chat():
    """Reset chat messages and session."""
    st.session_state.messages = []
    st.session_state.current_session_id = None


def start_new_chat():
    api = get_api()
    try:
        session = api.create_session()
        st.session_state.current_session_id = session["id"]
        st.session_state.messages = [
            {"role": "assistant",
             "content": "您好！我是 KnowFlow 知识库助手 🤖\n\n请上传文档或选择知识库后开始提问。"}
        ]
    except Exception:
        st.session_state.current_session_id = None
        reset_chat()


def load_kb_list():
    """Refresh KB list from API."""
    api = get_api()
    try:
        st.session_state.sidebar_kb_list = api.list_kbs()
        st.session_state.kbs_loaded = True
    except Exception:
        st.session_state.sidebar_kb_list = []
