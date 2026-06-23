"""
KnowFlow API Client

Encapsulates all backend API calls. Provides both sync (Streamlit) and async methods.
"""

import os
import json
from typing import Optional, List, Dict, Generator

import httpx

DEFAULT_BASE_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


class KnowFlowAPI:
    """HTTP client for all KnowFlow backend endpoints."""

    def __init__(self, base_url: str = None):
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self._client: Optional[httpx.Client] = None

    @property
    def client(self) -> httpx.Client:
        if self._client is None or self._client.is_closed:
            self._client = httpx.Client(timeout=60.0)
        return self._client

    def _get(self, path: str, **kw):
        """GET with retry on connection error."""
        try:
            r = self.client.get(self._url(path), **kw)
            r.raise_for_status()
            return r.json()
        except httpx.ConnectError:
            self._client = None  # Reset stale client
            r = self.client.get(self._url(path), **kw)
            r.raise_for_status()
            return r.json()

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    # ─── Health ───

    def health(self) -> dict:
        r = self.client.get(self._url("/health"))
        r.raise_for_status()
        return r.json()

    # ─── Knowledge Bases ───

    def list_kbs(self) -> List[dict]:
        return self._get("/api/v1/knowledge-bases")

    def create_kb(self, name: str, description: str = None,
                  kb_type: str = "document") -> dict:
        r = self.client.post(
            self._url("/api/v1/knowledge-bases"),
            json={"name": name, "description": description, "kb_type": kb_type},
        )
        r.raise_for_status()
        return r.json()

    def delete_kb(self, kb_id: str) -> dict:
        r = self.client.delete(
            self._url(f"/api/v1/knowledge-bases/{kb_id}")
        )
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
            r = self.client.post(
                self._url(f"/api/v1/knowledge-bases/{kb_id}/documents"),
                files=files,
                data=data,
                timeout=120.0,
            )
        r.raise_for_status()
        return r.json()

    def list_documents(self, kb_id: str) -> List[dict]:
        r = self.client.get(
            self._url(f"/api/v1/knowledge-bases/{kb_id}/documents")
        )
        r.raise_for_status()
        return r.json()

    def delete_document(self, kb_id: str, doc_id: str) -> dict:
        r = self.client.delete(
            self._url(f"/api/v1/knowledge-bases/{kb_id}/documents/{doc_id}")
        )
        r.raise_for_status()
        return r.json()

    # ─── Chat ───

    def chat(self, query: str, session_id: str = None,
             kb_ids: List[str] = None, enable_web_search: bool = False,
             enable_memory: bool = True, temperature: float = 0.1) -> dict:
        r = self.client.post(
            self._url("/api/v1/chat"),
            json={
                "query": query,
                "session_id": session_id,
                "kb_ids": kb_ids or [],
                "stream": False,
                "enable_web_search": enable_web_search,
                "enable_memory": enable_memory,
                "temperature": temperature,
            },
            timeout=120.0,
        )
        r.raise_for_status()
        return r.json()

    def chat_stream(self, query: str, session_id: str = None,
                    kb_ids: List[str] = None,
                    enable_web_search: bool = False,
                    enable_memory: bool = True,
                    temperature: float = 0.1) -> Generator[dict, None, None]:
        """Yield SSE events as parsed dicts."""
        with self.client.stream(
            "POST",
            self._url("/api/v1/chat/stream"),
            json={
                "query": query,
                "session_id": session_id,
                "kb_ids": kb_ids or [],
                "stream": True,
                "enable_web_search": enable_web_search,
                "enable_memory": enable_memory,
                "temperature": temperature,
            },
            timeout=120.0,
        ) as resp:
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
        r = self.client.get(self._url("/api/v1/conversations"))
        r.raise_for_status()
        return r.json()

    def create_session(self, title: str = "New Chat") -> dict:
        r = self.client.post(
            self._url("/api/v1/conversations"),
            json={"title": title},
        )
        r.raise_for_status()
        return r.json()

    def delete_session(self, session_id: str) -> dict:
        r = self.client.delete(
            self._url(f"/api/v1/conversations/{session_id}")
        )
        r.raise_for_status()
        return r.json()

    def get_messages(self, session_id: str) -> List[dict]:
        r = self.client.get(
            self._url(f"/api/v1/conversations/{session_id}/messages")
        )
        r.raise_for_status()
        return r.json()

    def close(self):
        if self._client:
            self._client.close()
            self._client = None
