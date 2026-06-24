"""
KnowFlow API Client — encapsulates all backend API calls with auth support.
"""

import os
import json
from typing import Optional, List, Dict, Generator

import httpx

DEFAULT_BASE_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


class KnowFlowAPI:
    """HTTP client with JWT auth."""

    def __init__(self, base_url: str = None, token: str = None):
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.token = token

    @property
    def _headers(self) -> dict:
        h = {}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _get(self, path: str) -> dict:
        with httpx.Client(timeout=60.0, headers=self._headers) as c:
            r = c.get(self._url(path))
            r.raise_for_status()
            return r.json()

    def _post(self, path: str, json_data: dict = None) -> dict:
        with httpx.Client(timeout=120.0, headers=self._headers) as c:
            r = c.post(self._url(path), json=json_data)
            r.raise_for_status()
            return r.json()

    # ─── Auth ───
    def login(self, email: str, password: str) -> dict:
        r = self._post("/api/v1/auth/login", {"email": email, "password": password})
        self.token = r.get("access_token")
        return r

    def register(self, username: str, email: str, password: str) -> dict:
        r = self._post("/api/v1/auth/register",
                       {"username": username, "email": email, "password": password})
        self.token = r.get("access_token")
        return r

    def logout(self):
        self.token = None

    # ─── Health ───
    def health(self) -> dict:
        return self._get("/health")

    # ─── Knowledge Bases ───
    def list_kbs(self) -> List[dict]:
        return self._get("/api/v1/knowledge-bases")

    def create_kb(self, name: str, description: str = None,
                  kb_type: str = "document", department: str = "_",
                  security_level: int = 1) -> dict:
        return self._post("/api/v1/knowledge-bases",
                          {"name": name, "description": description, "kb_type": kb_type,
                           "department": department, "security_level": security_level})

    def delete_kb(self, kb_id: str) -> dict:
        with httpx.Client(timeout=60.0, headers=self._headers) as c:
            r = c.delete(self._url(f"/api/v1/knowledge-bases/{kb_id}"))
            r.raise_for_status()
            return r.json()

    # ─── Documents ───
    def upload_document(self, kb_id: str, file_path: str,
                        title: str = None, original_name: str = None) -> dict:
        filename = original_name or os.path.basename(file_path)
        with open(file_path, "rb") as f:
            files = {"file": (filename, f)}
            data = {}
            if title:
                data["title"] = title
            with httpx.Client(timeout=120.0, headers=self._headers) as c:
                r = c.post(self._url(f"/api/v1/knowledge-bases/{kb_id}/documents"),
                           files=files, data=data)
                r.raise_for_status()
                return r.json()

    def list_documents(self, kb_id: str) -> List[dict]:
        return self._get(f"/api/v1/knowledge-bases/{kb_id}/documents")

    def delete_document(self, kb_id: str, doc_id: str) -> dict:
        with httpx.Client(timeout=60.0, headers=self._headers) as c:
            r = c.delete(self._url(f"/api/v1/knowledge-bases/{kb_id}/documents/{doc_id}"))
            r.raise_for_status()
            return r.json()

    # ─── Chat ───
    def chat_stream(self, query: str, session_id: str = None,
                    kb_ids: List[str] = None,
                    enable_web_search: bool = False,
                    enable_memory: bool = True,
                    temperature: float = 0.1) -> Generator[dict, None, None]:
        with httpx.Client(timeout=120.0, headers=self._headers) as c:
            with c.stream("POST", self._url("/api/v1/chat/stream"),
                          json={"query": query, "session_id": session_id,
                                "kb_ids": kb_ids or [], "stream": True,
                                "enable_web_search": enable_web_search,
                                "enable_memory": enable_memory,
                                "temperature": temperature}) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            yield json.loads(data_str)
                        except json.JSONDecodeError:
                            pass

    # ─── Conversations ───
    def list_sessions(self) -> List[dict]:
        return self._get("/api/v1/conversations")

    def create_session(self, title: str = "New Chat") -> dict:
        return self._post("/api/v1/conversations", {"title": title})

    def delete_session(self, session_id: str) -> dict:
        with httpx.Client(timeout=60.0, headers=self._headers) as c:
            r = c.delete(self._url(f"/api/v1/conversations/{session_id}"))
            r.raise_for_status()
            return r.json()

    def get_messages(self, session_id: str) -> List[dict]:
        return self._get(f"/api/v1/conversations/{session_id}/messages")
