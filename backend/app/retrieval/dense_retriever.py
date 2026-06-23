"""
Dense Vector retriever (Milvus)

Handles query embedding and vector similarity search.
Uses httpx directly to call DashScope compatible embedding API.
"""

import logging
from typing import List, Optional

import httpx

from app.core.config import settings
from app.retrieval.milvus_client import MilvusClient, SearchResult

logger = logging.getLogger(__name__)


class DenseRetriever:
    """Dense vector retriever backed by Milvus + DashScope embedding API."""

    def __init__(
        self,
        milvus: Optional[MilvusClient] = None,
        embedding_model: Optional[str] = None,
        embedding_api_key: Optional[str] = None,
        embedding_base_url: Optional[str] = None,
    ):
        self.milvus = milvus or MilvusClient()
        self.milvus.connect()

        self._model = embedding_model or settings.EMBEDDING_MODEL
        self._api_key = embedding_api_key or settings.EMBEDDING_API_KEY
        self._base_url = (embedding_base_url or settings.EMBEDDING_BASE_URL).rstrip("/")
        self._dimensions = settings.EMBEDDING_DIMENSION

    def _call_embedding_api(self, input_texts: List[str]) -> List[List[float]]:
        """Call DashScope compatible embeddings API via httpx."""
        url = f"{self._base_url}/embeddings"
        body = {
            "model": self._model,
            "input": input_texts,
            "dimensions": self._dimensions,
            "encoding_format": "float",
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        resp = httpx.post(url, json=body, headers=headers, timeout=60.0)
        if resp.status_code != 200:
            raise RuntimeError(f"Embedding API error {resp.status_code}: {resp.text[:500]}")
        data = resp.json()
        return [item["embedding"] for item in data["data"]]

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query text."""
        return self.embed_documents([query])[0]

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Batch embed documents. Filters empty strings."""
        docs = [d for d in documents if d and isinstance(d, str) and d.strip()]
        if not docs:
            logger.warning("embed_documents: all empty, returning []")
            return []

        try:
            embeddings = self._call_embedding_api(docs)
            logger.debug("Embedded %d docs, dim=%d", len(embeddings),
                          len(embeddings[0]) if embeddings else 0)
            return embeddings
        except Exception as e:
            logger.error("Batch embedding failed (%d docs): %s", len(docs), e)
            raise

    def search(
        self,
        query: str,
        kb_ids: Optional[List[str]] = None,
        top_k: int = 50,
        threshold: float = 0.15,
        query_embedding: Optional[List[float]] = None,
    ) -> List[SearchResult]:
        """Execute dense vector search."""
        if query_embedding is None:
            query_embedding = self.embed_query(query)

        results = self.milvus.search(
            query_embedding=query_embedding,
            kb_ids=kb_ids,
            top_k=top_k,
            threshold=threshold,
        )
        logger.info(
            "Dense search: query='%s...', kbs=%s, results=%d",
            query[:50], kb_ids, len(results),
        )
        return results
