"""
Milvus 向量数据库客户端

提供 Collection 管理、向量索引创建、增删查等基础操作。
每个知识库对应 Milvus 中的一个 Partition，实现逻辑隔离。

Milvus 版本要求: >= 2.5.0
"""

import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
    Partition,
)
from pymilvus.client.types import LoadState

from app.core.config import settings

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# Milvus Schema 定义
# ═══════════════════════════════════════════════════════════

# Collection 主键字段
PRIMARY_FIELD = "pk_id"           # 自增主键
# 向量字段
VECTOR_FIELD = "embedding"        # 向量 (FloatVector)
# 标量字段（用于过滤和关联）
CHUNK_ID_FIELD = "chunk_id"       # Chunk UUID（与 PostgreSQL 关联）
KB_ID_FIELD = "kb_id"             # 知识库 ID（用于 Partition 过滤）
DOC_ID_FIELD = "doc_id"           # 文档 ID
CONTENT_FIELD = "content"         # 原始文本（大字段）

# 向量维度
VECTOR_DIM = settings.EMBEDDING_DIMENSION

# 默认索引参数
DEFAULT_INDEX_PARAMS = {
    "index_type": "IVF_FLAT",         # 倒排索引 + 精确搜索
    "metric_type": "IP",              # Inner Product（等价于 Cosine 对归一化向量）
    "params": {"nlist": 1024},        # 聚类数
}


@dataclass
class SearchResult:
    """单条检索结果"""
    chunk_id: str
    content: str
    score: float
    kb_id: str = ""
    doc_id: str = ""


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
        self._connected = False
        self._collection: Optional[Collection] = None
        self._collection_name = settings.MILVUS_COLLECTION_NAME

    # ─── 连接管理 ───
    def connect(self):
        """建立 Milvus 连接"""
        if self._connected:
            return

        connections.connect(
            alias="default",
            host=settings.MILVUS_HOST,
            port=str(settings.MILVUS_PORT),
        )
        self._connected = True
        logger.info(f"Connected to Milvus at {settings.MILVUS_HOST}:{settings.MILVUS_PORT}")

    def disconnect(self):
        """断开 Milvus 连接"""
        if self._connected:
            connections.disconnect("default")
            self._connected = False

    # ─── Collection 管理 ───
    @property
    def collection(self) -> Collection:
        """获取（懒加载）Collection 对象"""
        if self._collection is None:
            self.connect()
            if not utility.has_collection(self._collection_name):
                self._create_collection_internal()
            self._collection = Collection(self._collection_name)
            self._collection.load()
        return self._collection

    def _create_collection_internal(self):
        """内部：创建 Collection"""
        fields = [
            FieldSchema(
                name=PRIMARY_FIELD,
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=True,
            ),
            FieldSchema(
                name=VECTOR_FIELD,
                dtype=DataType.FLOAT_VECTOR,
                dim=VECTOR_DIM,
            ),
            FieldSchema(
                name=CHUNK_ID_FIELD,
                dtype=DataType.VARCHAR,
                max_length=64,
            ),
            FieldSchema(
                name=KB_ID_FIELD,
                dtype=DataType.VARCHAR,
                max_length=64,
            ),
            FieldSchema(
                name=DOC_ID_FIELD,
                dtype=DataType.VARCHAR,
                max_length=64,
            ),
            FieldSchema(
                name=CONTENT_FIELD,
                dtype=DataType.VARCHAR,
                max_length=65535,  # 文本可能很长
            ),
        ]

        schema = CollectionSchema(
            fields=fields,
            description="KnowFlow knowledge base vectors",
            enable_dynamic_field=False,
        )

        self._collection = Collection(
            name=self._collection_name,
            schema=schema,
        )
        logger.info(f"Created Milvus collection: {self._collection_name}")

        # 创建向量索引
        self._collection.create_index(
            field_name=VECTOR_FIELD,
            index_params=DEFAULT_INDEX_PARAMS,
        )
        logger.info(f"Created vector index: {DEFAULT_INDEX_PARAMS}")

        # 加载到内存
        self._collection.load()

    def create_collection(self, drop_existing: bool = False):
        """
        创建 Collection（公开接口）

        Args:
            drop_existing: 是否删除已存在的 Collection 后重建
        """
        self.connect()

        if utility.has_collection(self._collection_name):
            if drop_existing:
                utility.drop_collection(self._collection_name)
                logger.info(f"Dropped existing collection: {self._collection_name}")
            else:
                logger.info(f"Collection already exists: {self._collection_name}")
                self._collection = Collection(self._collection_name)
                self._collection.load()
                return

        self._create_collection_internal()

    # ─── Partition 管理 ───
    def ensure_partition(self, kb_id: str):
        """确保知识库对应的 Partition 存在"""
        if not self.collection.has_partition(kb_id):
            self.collection.create_partition(kb_id)
            logger.info(f"Created partition for KB: {kb_id}")

    def drop_partition(self, kb_id: str):
        """删除知识库对应的 Partition（级联删除所有向量）"""
        if self.collection.has_partition(kb_id):
            self.collection.release()
            self.collection.drop_partition(kb_id)
            self.collection.load()
            logger.info(f"Dropped partition for KB: {kb_id}")

    # ─── 向量插入 ───
    def insert_vectors(
        self,
        chunk_ids: List[str],
        embeddings: List[List[float]],
        contents: List[str],
        kb_id: str,
        doc_ids: Optional[List[str]] = None,
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

        self.ensure_partition(kb_id)

        # 构建插入数据
        data = [
            embeddings,                    # FloatVector
            chunk_ids,                     # VARCHAR chunk_id
            [kb_id] * n,                   # VARCHAR kb_id
            doc_ids,                       # VARCHAR doc_id
            contents,                      # VARCHAR content
        ]

        try:
            mr = self.collection.insert(data, partition_name=kb_id)
            inserted_ids = mr.primary_keys
            logger.info(
                f"Inserted {len(inserted_ids)} vectors into KB '{kb_id}' "
                f"(partition: {kb_id})"
            )
            return inserted_ids
        except Exception as e:
            logger.error(f"Failed to insert vectors: {e}")
            raise

    # ─── 向量检索 ───
    def search(
        self,
        query_embedding: List[float],
        kb_ids: Optional[List[str]] = None,
        top_k: int = 50,
        threshold: float = 0.15,
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
        search_params = {
            "metric_type": "IP",
            "params": {"nprobe": 16},  # 搜索的聚类数
        }

        # 构建过滤表达式（按知识库过滤）
        expr = None
        if kb_ids and len(kb_ids) > 0:
            if len(kb_ids) == 1:
                expr = f'{KB_ID_FIELD} == "{kb_ids[0]}"'
            else:
                kb_list = ', '.join(f'"{kb}"' for kb in kb_ids)
                expr = f'{KB_ID_FIELD} in [{kb_list}]'

        # 搜索的 Partition 列表
        partitions = kb_ids if kb_ids else None

        try:
            results = self.collection.search(
                data=[query_embedding],
                anns_field=VECTOR_FIELD,
                param=search_params,
                limit=top_k,
                expr=expr,
                partition_names=partitions,
                output_fields=[
                    CHUNK_ID_FIELD,
                    KB_ID_FIELD,
                    DOC_ID_FIELD,
                    CONTENT_FIELD,
                ],
            )

            # 格式化结果
            formatted = []
            for hits in results:
                for hit in hits:
                    if hit.score >= threshold:
                        formatted.append(SearchResult(
                            chunk_id=hit.entity.get(CHUNK_ID_FIELD),
                            content=hit.entity.get(CONTENT_FIELD, ""),
                            score=float(hit.score),
                            kb_id=hit.entity.get(KB_ID_FIELD, ""),
                            doc_id=hit.entity.get(DOC_ID_FIELD, ""),
                        ))

            logger.debug(
                f"Vector search returned {len(formatted)} results "
                f"(top_k={top_k}, threshold={threshold})"
            )
            return formatted

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            raise

    # ─── 删除操作 ───
    def delete_by_chunk_ids(self, chunk_ids: List[str], kb_id: str):
        """按 Chunk ID 删除向量"""
        if not chunk_ids:
            return
        id_list = ', '.join(f'"{cid}"' for cid in chunk_ids)
        expr = f'{CHUNK_ID_FIELD} in [{id_list}]'
        self.collection.delete(expr, partition_name=kb_id)
        logger.info(f"Deleted {len(chunk_ids)} vectors from KB '{kb_id}'")

    def delete_by_kb_id(self, kb_id: str):
        """删除某个知识库的所有向量"""
        self.collection.delete(
            f'{KB_ID_FIELD} == "{kb_id}"',
            partition_name=kb_id,
        )
        logger.info(f"Deleted all vectors from KB '{kb_id}'")

    # ─── 统计信息 ───
    def get_collection_stats(self) -> Dict:
        """获取 Collection 统计信息"""
        stats = self.collection.num_entities
        return {
            "collection_name": self._collection_name,
            "total_entities": stats,
        }

    def get_kb_count(self, kb_id: str) -> int:
        """获取某个知识库的向量数量（近似值）"""
        self.ensure_partition(kb_id)
        partition = Partition(self.collection, kb_id)
        return partition.num_entities


# ─── 全局单例 ───
milvus_client = MilvusClient()
