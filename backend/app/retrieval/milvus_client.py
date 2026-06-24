"""
Milvus 向量数据库客户端

提供 Collection 管理、向量索引创建、增删查等基础操作。
每个知识库对应 Milvus 中的一个 Partition，实现逻辑隔离。

Milvus 版本要求: >= 2.5.0
使用 PyMilvus 新式 MilvusClient API (connections.connect 已废弃)。
"""

import logging
from typing import List, Dict, Optional
from dataclasses import dataclass

from pymilvus import MilvusClient as PyMilvusClient
from pymilvus import DataType

from app.core.config import settings

logger = logging.getLogger(__name__)


# ============================================================
# Milvus Schema 定义
# ============================================================

# Collection 主键字段
PRIMARY_FIELD = "pk_id"           # 自增主键
# 向量字段
VECTOR_FIELD = "embedding"
CHUNK_ID_FIELD = "chunk_id"
KB_ID_FIELD = "kb_id"
DOC_ID_FIELD = "doc_id"
CONTENT_FIELD = "content"
SECURITY_FIELD = "security_level"   # RBAC pre-filter
DEPT_FIELD = "department"           # RBAC pre-filter

VECTOR_DIM = settings.EMBEDDING_DIMENSION


@dataclass
class SearchResult:
    """Search result with document metadata."""
    chunk_id: str
    content: str
    score: float
    kb_id: str = ""
    doc_id: str = ""
    doc_title: str = ""
    doc_filename: str = ""


class MilvusClient:
    """
    Milvus 向量数据库客户端

    负责：
    - 连接管理
    - Collection 生命周期（创建/加载/释放/删除）
    - 向量插入与检索
    - Partition 管理（每个知识库一个 Partition）

    用法:
        client = MilvusClient()
        client.connect()
        client.create_collection()
        client.insert_vectors(chunks, embeddings, kb_id)
        results = client.search(query_embedding, kb_ids=["kb_1", "kb_2"])
    """

    def __init__(self):
        self._client: Optional[PyMilvusClient] = None
        self._collection_name = settings.MILVUS_COLLECTION_NAME
        self._collection_loaded = False

    # ─── 连接管理 ───

    def connect(self):
        """建立 Milvus 连接（新式 API）"""
        if self._client is not None:
            return

        self._client = PyMilvusClient(
            uri=f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}",
        )
        logger.info(
            "Connected to Milvus at %s:%s",
            settings.MILVUS_HOST, settings.MILVUS_PORT,
        )

    def disconnect(self):
        """断开 Milvus 连接"""
        if self._client is not None:
            self._client.close()
            self._client = None
            self._collection_loaded = False

    # ─── Collection 管理 ───

    def _ensure_loaded(self):
        """Ensure collection exists and is loaded."""
        self.connect()
        if self._collection_loaded:
            return
        if not self._client.has_collection(self._collection_name):
            self._create_collection_internal()
            return
        try:
            self._client.load_collection(self._collection_name)
            self._collection_loaded = True
        except Exception as e:
            logger.warning("Load collection failed, will retry: %s", e)
            self._collection_loaded = False

    def _create_collection_internal(self):
        """Create Milvus collection with custom schema."""
        schema = PyMilvusClient.create_schema(enable_dynamic_field=False)

        # add_field signature: (field_name, datatype, **kwargs)
        schema.add_field(PRIMARY_FIELD, DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(VECTOR_FIELD, DataType.FLOAT_VECTOR, dim=VECTOR_DIM)
        schema.add_field(CHUNK_ID_FIELD, DataType.VARCHAR, max_length=64)
        schema.add_field(KB_ID_FIELD, DataType.VARCHAR, max_length=64)
        schema.add_field(DOC_ID_FIELD, DataType.VARCHAR, max_length=64)
        schema.add_field(CONTENT_FIELD, DataType.VARCHAR, max_length=65535)
        schema.add_field(SECURITY_FIELD, DataType.INT64)  # 0-3 clearance
        schema.add_field(DEPT_FIELD, DataType.VARCHAR, max_length=64)

        index_params = PyMilvusClient.prepare_index_params()
        index_params.add_index(
            field_name=VECTOR_FIELD,
            index_type="IVF_FLAT",
            metric_type="IP",
            params={"nlist": 1024},
        )

        try:
            self._client.create_collection(
                collection_name=self._collection_name,
                schema=schema,
                index_params=index_params,
            )
            logger.info("Created Milvus collection: %s", self._collection_name)
        except Exception as e:
            logger.error("Failed to create collection with schema: %s", e)
            # Fallback: simple collection without custom fields
            self._client.drop_collection(self._collection_name)
            self._client.create_collection(
                collection_name=self._collection_name,
                dimension=VECTOR_DIM,
                metric_type="IP",
            )
            logger.info("Created Milvus collection (simple): %s", self._collection_name)

        # Load after creation
        import time
        time.sleep(0.5)
        self._client.load_collection(self._collection_name)
        self._collection_loaded = True

    def create_collection(self, drop_existing: bool = False):
        """
        创建 Collection（公开接口）

        Args:
            drop_existing: 是否删除已存在的 Collection 后重建
        """
        self.connect()

        if self._client.has_collection(self._collection_name):
            if drop_existing:
                self._client.drop_collection(self._collection_name)
                logger.info(
                    "Dropped existing collection: %s", self._collection_name
                )
            else:
                logger.info(
                    "Collection already exists: %s", self._collection_name
                )
                self._ensure_loaded()
                return

        self._create_collection_internal()

    # ─── Partition 管理 ───

    def _ensure_partition(self, kb_id: str):
        """确保知识库对应的 Partition 存在"""
        self._ensure_loaded()
        if not self._client.has_partition(self._collection_name, kb_id):
            self._client.create_partition(self._collection_name, kb_id)
            logger.info("Created partition for KB: %s", kb_id)

    def drop_partition(self, kb_id: str):
        """删除知识库对应的 Partition（级联删除所有向量）"""
        self._ensure_loaded()
        if self._client.has_partition(self._collection_name, kb_id):
            self._client.drop_partition(self._collection_name, kb_id)
            logger.info("Dropped partition for KB: %s", kb_id)

    # ─── 向量插入 ───

    def insert_vectors(
        self,
        chunk_ids: List[str],
        embeddings: List[List[float]],
        contents: List[str],
        kb_id: str,
        doc_ids: Optional[List[str]] = None,
        security_levels: Optional[List[int]] = None,
        departments: Optional[List[str]] = None,
    ) -> List[int]:
        """
        批量插入向量

        Args:
            chunk_ids: Chunk ID 列表
            embeddings: 对应的向量列表
            contents: 对应的文本内容列表
            kb_id: 知识库 ID
            doc_ids: 文档 ID 列表（可选）

        Returns:
            插入成功的向量 ID 列表

        Raises:
            ValueError: 输入长度不一致
        """
        n = len(chunk_ids)
        if len(embeddings) != n or len(contents) != n:
            raise ValueError(
                f"Input length mismatch: chunk_ids={len(chunk_ids)}, "
                f"embeddings={len(embeddings)}, contents={len(contents)}"
            )

        if doc_ids is None:
            doc_ids = [""] * n

        self._ensure_partition(kb_id)

        # 构建插入数据 — MilvusClient 使用 list-of-dicts 格式
        data = []
        for i in range(n):
            data.append({
                VECTOR_FIELD: embeddings[i],
                CHUNK_ID_FIELD: chunk_ids[i],
                KB_ID_FIELD: kb_id,
                DOC_ID_FIELD: doc_ids[i],
                CONTENT_FIELD: contents[i],
                SECURITY_FIELD: security_levels[i] if security_levels else 1,
                DEPT_FIELD: departments[i] if departments else "",
            })

        try:
            result = self._client.insert(
                collection_name=self._collection_name,
                data=data,
                partition_name=kb_id,
            )
            inserted_ids = result["ids"]
            logger.info(
                "Inserted %d vectors into KB '%s' (partition: %s)",
                len(inserted_ids), kb_id, kb_id,
            )
            return inserted_ids
        except Exception as e:
            logger.error("Failed to insert vectors: %s", e)
            raise

    # ─── 向量检索 ───

    def search(
        self,
        query_embedding: List[float],
        kb_ids: Optional[List[str]] = None,
        top_k: int = 50,
        threshold: float = 0.15,
        permission_expr: str = "",
    ) -> List[SearchResult]:
        """
        向量相似度检索

        Args:
            query_embedding: 查询向量
            kb_ids: 知识库 ID 列表（None = 搜索所有知识库）
            top_k: 返回数量
            threshold: 最低分数阈值

        Returns:
            检索结果列表（按分数降序）
        """
        self._ensure_loaded()

        search_params = {
            "metric_type": "IP",
            "params": {"nprobe": 16},
        }

        # Build filter: KB filter + permission pre-filter
        parts = []
        if kb_ids and len(kb_ids) > 0:
            if len(kb_ids) == 1:
                parts.append(f'{KB_ID_FIELD} == "{kb_ids[0]}"')
            else:
                kb_list = ", ".join(f'"{kb}"' for kb in kb_ids)
                parts.append(f"{KB_ID_FIELD} in [{kb_list}]")
        if permission_expr:
            parts.append(f"({permission_expr})")
        filter_expr = " AND ".join(parts) if parts else ""

        try:
            # MilvusClient.search 返回 List[List[dict]]
            # 外层 list 对应每个查询向量，内层 list 是每条命中
            results = self._client.search(
                collection_name=self._collection_name,
                data=[query_embedding],
                filter=filter_expr,
                limit=top_k,
                output_fields=[
                    CHUNK_ID_FIELD,
                    KB_ID_FIELD,
                    DOC_ID_FIELD,
                    CONTENT_FIELD,
                ],
                search_params=search_params,
                partition_names=kb_ids if kb_ids else None,
                anns_field=VECTOR_FIELD,
            )

            # 格式化结果
            formatted = []
            for hits in results:
                for hit in hits:
                    if hit["distance"] >= threshold:
                        entity = hit.get("entity", {})
                        formatted.append(SearchResult(
                            chunk_id=entity.get(CHUNK_ID_FIELD, ""),
                            content=entity.get(CONTENT_FIELD, ""),
                            score=float(hit["distance"]),
                            kb_id=entity.get(KB_ID_FIELD, ""),
                            doc_id=entity.get(DOC_ID_FIELD, ""),
                        ))

            logger.debug(
                "Vector search returned %d results (top_k=%d, threshold=%.2f)",
                len(formatted), top_k, threshold,
            )
            return formatted

        except Exception as e:
            logger.error("Vector search failed: %s", e)
            raise

    # ─── 删除操作 ───

    def delete_by_chunk_ids(self, chunk_ids: List[str], kb_id: str):
        """按 Chunk ID 删除向量"""
        if not chunk_ids:
            return
        self._ensure_loaded()
        id_list = ", ".join(f'"{cid}"' for cid in chunk_ids)
        filter_expr = f"{CHUNK_ID_FIELD} in [{id_list}]"
        self._client.delete(
            collection_name=self._collection_name,
            filter=filter_expr,
            partition_name=kb_id,
        )
        logger.info(
            "Deleted %d vectors from KB '%s'", len(chunk_ids), kb_id,
        )

    def delete_by_kb_id(self, kb_id: str):
        """删除某个知识库的所有向量"""
        self._ensure_loaded()
        self._client.delete(
            collection_name=self._collection_name,
            filter=f'{KB_ID_FIELD} == "{kb_id}"',
            partition_name=kb_id,
        )
        logger.info("Deleted all vectors from KB '%s'", kb_id)

    # ─── 统计信息 ───

    def get_collection_stats(self) -> Dict:
        """获取 Collection 统计信息"""
        self._ensure_loaded()
        stats = self._client.get_collection_stats(self._collection_name)
        return {
            "collection_name": self._collection_name,
            "total_entities": stats.get("row_count", 0),
        }

    def get_kb_count(self, kb_id: str) -> int:
        """获取某个知识库的向量数量"""
        self._ensure_partition(kb_id)
        try:
            results = self._client.query(
                collection_name=self._collection_name,
                filter=f'{KB_ID_FIELD} == "{kb_id}"',
                output_fields=[KB_ID_FIELD],
                partition_names=[kb_id],
            )
            return len(results)
        except Exception:
            return 0


# ─── 全局单例 ───
milvus_client = MilvusClient()
