# 🤖 KnowFlow — 企业级 RAG 知识库问答助手

<p align="center">
  <strong>LangGraph + Milvus 驱动的多租户 RAG 知识工作流引擎</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.115+-green.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/LangGraph-0.4+-orange.svg" alt="LangGraph">
  <img src="https://img.shields.io/badge/Milvus-2.5+-00BFFF.svg" alt="Milvus">
  <img src="https://img.shields.io/badge/Streamlit-1.40+-red.svg" alt="Streamlit">
  <img src="https://img.shields.io/badge/PostgreSQL-16+-blue.svg" alt="PostgreSQL">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

---

## 📌 项目简介

KnowFlow 是一款生产级 **多租户 RAG 智能问答系统**，基于 LangChain + LangGraph + Milvus 构建，面向企业知识管理场景。

### 核心能力

- 🔍 **混合检索** — Milvus 向量检索 + BM25 关键词检索，动态 RRF 融合，Qwen3-Rerank 精排
- 🧠 **记忆压缩** — 多轮对话 LLM 语义压缩，Token 感知触发
- 🛡️ **RBAC 多租户** — 用户角色/部门/密级权限，Milvus Pre-filter 强制注入
- 🎯 **自适应分块** — Heading / Heuristic / Recursive 三层策略 + Parent-Child 模式
- 📊 **Langfuse 可观测** — 全链路 Trace：检索→Rerank→LLM 生成
- ⚡ **异步文档处理** — Celery 后台解析、分块、向量化
- 🖼️ **多格式支持** — PDF / Word / PPT / Excel / Markdown / TXT / CSV / 图片 OCR
- 🔐 **JWT 认证** — 注册/登录，所有 API 请求鉴权
- 🚦 **速率限制** — 100/分钟每用户，300/分钟每 IP

---

## 🏗️ 系统架构

```
Streamlit UI ──→ FastAPI ──→ LangGraph Pipeline ──→ Milvus + PostgreSQL + Redis
    │               │              │
    │               ├─ Auth (JWT)  ├─ Query Rewrite
    │               ├─ RBAC        ├─ Hybrid Search (Dense + BM25)
    │               ├─ Rate Limit  ├─ Dynamic RRF Fusion
    │               └─ Celery      ├─ Qwen3 Rerank
    │                              ├─ Memory Consolidator
    │                              └─ LLM Generation (DeepSeek)
    │
    └── 多格式文档上传 → Celery Worker → 解析 → 分块 → Embedding → Milvus + BM25
```

---

## 🚀 快速开始

### 环境要求

- Python 3.11+
- Docker & Docker Compose
- Git

### 本地开发

```bash
# 1. 克隆项目
git clone https://github.com/LLIEsmallwhite/KnowFlow.git
cd KnowFlow

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env，填入 LLM API Key 和 Embedding API Key

# 3. 安装依赖
cd backend && pip install -r requirements.txt
cd ../frontend && pip install -r requirements.txt
cd ..

# 4. 启动基础设施
docker compose up -d postgres redis milvus-standalone minio etcd

# 5. 启动后端
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 6. （可选）启动 Celery 异步文档处理
celery -A tasks.document_tasks worker --loglevel=info --concurrency=4 -P solo

# 7. 启动前端
cd frontend
streamlit run app.py
```

访问：

| 服务 | 地址 |
|------|------|
| 前端 UI | http://localhost:8501 |
| API 文档 | http://localhost:8000/docs |
| MinIO 控制台 | http://localhost:9001 |

---

## 📁 项目结构

```
KnowFlow/
├── backend/
│   ├── app/
│   │   ├── api/              # FastAPI 路由
│   │   │   ├── auth.py       # 注册/登录
│   │   │   ├── chat.py       # RAG 对话 (SSE 流式)
│   │   │   ├── agent.py      # Agent ReAct 对话
│   │   │   ├── knowledge_base.py  # 知识库 CRUD + 文档管理
│   │   │   └── conversation.py    # 会话历史
│   │   ├── core/             # 基础设施
│   │   │   ├── config.py     # Pydantic Settings
│   │   │   ├── database.py   # SQLAlchemy async engine
│   │   │   ├── auth.py       # JWT + bcrypt
│   │   │   ├── dependencies.py  # 依赖注入 (get_current_user)
│   │   │   ├── permissions.py   # RBAC 权限引擎
│   │   │   └── rate_limit.py    # 令牌桶限流
│   │   ├── models/           # SQLAlchemy ORM
│   │   │   ├── user.py       # 用户 (角色/密级/部门)
│   │   │   ├── knowledge_base.py  # 知识库 (安全标签)
│   │   │   ├── document.py   # 文档
│   │   │   ├── chunk.py      # 文本片段 (含安全字段)
│   │   │   ├── session.py    # 对话会话
│   │   │   └── message.py    # 对话消息
│   │   ├── retrieval/        # ⭐ 检索核心
│   │   │   ├── milvus_client.py     # Milvus 客户端
│   │   │   ├── dense_retriever.py   # Dense 向量检索
│   │   │   ├── bm25_retriever.py    # BM25 关键词检索
│   │   │   ├── hybrid_search.py     # 混合检索编排器
│   │   │   ├── dynamic_rrf.py       # 动态 RRF 融合
│   │   │   ├── dedup.py             # 多级去重
│   │   │   ├── reranker.py          # Qwen3-Rerank / Cross-Encoder
│   │   │   └── context_merge.py     # 上下文合并
│   │   ├── memory/           # 记忆管理
│   │   │   ├── consolidator.py      # LLM 记忆压缩
│   │   │   └── token_estimator.py   # Token 估算
│   │   ├── graph/            # LangGraph 编排
│   │   │   ├── states.py            # State 定义
│   │   │   ├── rag_pipeline.py      # RAG 7 节点 Pipeline
│   │   │   └── agent_graph.py       # Agent ReAct Graph
│   │   ├── utils/            # 工具
│   │   │   ├── chunking.py          # 自适应三层分块
│   │   │   ├── document_loaders.py  # 多格式加载器 (含 OCR)
│   │   │   └── text_processing.py   # 文本归一化
│   │   ├── services/         # 业务层
│   │   │   ├── doc_service.py       # 文档处理流水线
│   │   │   ├── kb_crud.py           # 知识库 CRUD
│   │   │   ├── document_crud.py     # 文档 CRUD
│   │   │   ├── chunk_crud.py        # Chunk CRUD
│   │   │   ├── session_crud.py      # 会话 CRUD
│   │   │   ├── message_crud.py      # 消息 CRUD
│   │   │   └── minio_service.py     # 对象存储
│   │   ├── agent/
│   │   │   └── tools.py             # Agent 工具注册
│   │   └── observability/
│   │       └── langfuse_client.py   # Langfuse 追踪
│   ├── eval/                 # RAG 评测
│   │   └── evaluate.py             # 检索质量评测脚本
│   ├── tasks/                # Celery 异步
│   │   └── document_tasks.py       # 文档后台处理
│   ├── migrations/           # Alembic 迁移
│   ├── requirements.txt
│   └── main.py               # FastAPI 入口
├── frontend/
│   ├── pages/                # Streamlit 多页面
│   │   ├── 1_Knowledge_Base.py     # 知识库管理
│   │   └── 2_Conversations.py      # 对话历史
│   ├── components/           # UI 组件
│   │   ├── sidebar.py              # 侧边栏 (登录/KB/设置)
│   │   └── search_viz.py           # 检索可视化
│   ├── utils/                # 前端工具
│   │   ├── api.py                  # API 客户端 (JWT)
│   │   └── session.py              # 会话状态管理
│   ├── app.py                # Streamlit 入口
│   └── requirements.txt
├── sample_docs/              # 测试文档 (已 gitignore)
├── docker-compose.yml        # 完整容器编排
├── .env.example              # 环境变量模板
└── README.md
```

---

## ⭐ 核心特性详解

### RBAC 多租户权限

```
User (role/clearance/departments)
  │
  └── PermissionFilter ──→ Milvus Pre-filter
         "security_level <= 2 AND department in ['engineering','product']"
```

- 三角色：admin（全量）、manager（本部门+低密级）、member（本部门公开+内部）
- 四密级：公开(0) / 内部(1) / 机密(2) / 绝密(3)
- 文档创建时继承 KB 安全标签，Chunk 降级存储用于向量检索过滤
- 向量检索阶段注入权限断言，确保召回结果 100% 合规

### 动态 RRF 融合

基于四因子自适应计算向量/关键词权重：

1. **查询类型** — 精确查询（代码/ID）→ 提升 Keyword；概念查询 → 提升 Vector
2. **查询长度** — 短查询语义信号弱 → 提升 Keyword
3. **结果分布** — 某路检索结果极少 → 补偿另一路
4. **分数方差** — 方差大 = 区分度高 → 加权

### 自适应三层分块

```
Structural → Semantic → Fixed
    ↓            ↓         ↓
 标题/代码块   Embedding  固定大小
   边界        相似度      兜底
```

Parent-Child 模式：子块用于检索（~384 tokens）、父块返回给 LLM（~4096 tokens）。

### 异步文档处理

```
上传 → 立即返回 "processing" → Celery Worker
                                  ├── 解析文档
                                  ├── 自适应分块
                                  ├── Embedding (DashScope, batch=10)
                                  ├── Milvus 写入 (含安全标签)
                                  └── BM25 索引构建
```

### 可观测性 (Langfuse)

启用后自动追踪：检索统计、Rerank 结果、LLM Token 消耗、端到端延迟。

---

## 🔧 技术栈

| 层级 | 技术 |
|------|------|
| Web 框架 | FastAPI + Uvicorn |
| LLM 编排 | LangChain + LangGraph |
| 对话模型 | DeepSeek v4-Pro |
| Embedding | Qwen text-embedding-v3 (DashScope) |
| Rerank | Qwen3-Rerank (DashScope) |
| 向量数据库 | Milvus 2.5 |
| 关系数据库 | PostgreSQL 16 |
| 缓存/消息队列 | Redis 7 + Celery |
| 对象存储 | MinIO |
| 前端 | Streamlit 1.40+ |
| 可观测性 | Langfuse |
| OCR | Qwen-VL-Max (DashScope) |

---

## 📊 API 端点总览

| 分组 | 端点 | 说明 |
|------|------|------|
| 认证 | POST `/auth/register` | 注册 |
| | POST `/auth/login` | 登录 |
| | GET `/auth/me` | 当前用户 |
| 知识库 | GET/POST `/knowledge-bases` | 列表/创建 |
| | GET/PATCH/DELETE `/{kb_id}` | 详情/更新/删除 |
| | POST `/{kb_id}/documents` | 上传文档 |
| | GET/DELETE `/{kb_id}/documents/{id}` | 文档详情/删除 |
| | GET `/{kb_id}/documents/{id}/status` | 处理状态轮询 |
| | POST `/search` | 检索测试 |
| 对话 | POST `/chat` | 同步问答 |
| | POST `/chat/stream` | SSE 流式问答 |
| Agent | POST `/agent/chat` | Agent 问答 |
| 会话 | GET/POST/DELETE `/conversations` | 会话管理 |
| | GET `/conversations/{id}/messages` | 消息列表 |
| 调试 | GET `/debug/bm25` | BM25 状态 |
| | GET `/debug/chunks-status` | Chunk 向量化状态 |
| | POST `/debug/rebuild-bm25` | 重建 BM25 |
| | POST `/debug/revectorize` | 重新向量化 |
| 系统 | GET `/health` | 健康检查 |

---

## 🚧 未来规划

### 近期

- [ ] **Elasticsearch 关键词检索引擎** — 替代内存 BM25，支持亿级 Chunk
- [ ] **前端文档状态实时轮询** — 上传后自动显示处理进度
- [ ] **图片多模态问答** — 上传图片直接提问，不经过 OCR
- [ ] **Web 端管理后台** — 用户管理、KB 权限配置、审计日志查看

### 中期

- [ ] **多轮对话意图感知 Rerank** — 结合对话历史对候选 Chunk 二次打分
- [ ] **混合分块策略自适应** — 根据文档类型自动选择最优分块算法
- [ ] **Milvus IVF 索引升级** — IVF_SQ8 量化索引，内存节省 75%
- [ ] **GraphRAG 知识图谱增强** — 实体关系抽取 + 图检索

### 远期

- [ ] **多模态文档理解** — 图片/表格/公式统一 Embedding
- [ ] **Agent 工具生态** — Web Search / SQL Query / Code Interpreter
- [ ] **联邦知识库** — 跨组织知识共享，隐私计算
- [ ] **Kubernetes 生产部署** — Helm Chart + HPA 自动扩缩 + Istio 服务网格

---

## 📄 License

MIT License

---

<p align="center">
  Built with ❤️ by <a href="https://github.com/LLIEsmallwhite">LLIEsmallwhite</a>
</p>
