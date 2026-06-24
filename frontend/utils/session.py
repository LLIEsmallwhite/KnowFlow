"""
Session State Manager for Streamlit — persists auth across refreshes.
"""

import os
import json
import streamlit as st
from utils.api import KnowFlowAPI

TOKEN_FILE = os.path.join(os.path.dirname(__file__), "..", ".streamlit", "auth.json")


def _load_auth():
    try:
        with open(TOKEN_FILE) as f:
            d = json.load(f)
            return d.get("token", ""), d.get("username", "")
    except Exception:
        return "", ""


def _save_auth(token: str, username: str):
    try:
        os.makedirs(os.path.dirname(TOKEN_FILE), exist_ok=True)
        with open(TOKEN_FILE, "w") as f:
            json.dump({"token": token, "username": username}, f)
    except Exception:
        pass


def init_session_state():
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
    }

    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # Restore persisted auth token
    if "auth_token" not in st.session_state:
        token, username = _load_auth()
        st.session_state.auth_token = token
        st.session_state.username = username
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = bool(st.session_state.auth_token)
    if "username" not in st.session_state:
        st.session_state.username = ""

    # Always init API client
    if st.session_state.api is None:
        st.session_state.api = KnowFlowAPI(token=st.session_state.auth_token)

    # Validate persisted token
    if st.session_state.logged_in:
        try:
            st.session_state.api.health()
        except Exception:
            st.session_state.logged_in = False
            st.session_state.auth_token = None
            _save_token("")
            st.session_state.api = KnowFlowAPI()


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
