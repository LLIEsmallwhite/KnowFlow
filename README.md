# 🤖 KnowFlow — 企业级 RAG 知识库问答助手

<p align="center">
  <strong>Knowledge + Workflow = LangGraph 驱动的知识工作流引擎</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.115+-green.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/LangGraph-0.4+-orange.svg" alt="LangGraph">
  <img src="https://img.shields.io/badge/Milvus-2.5+-00BFFF.svg" alt="Milvus">
  <img src="https://img.shields.io/badge/Streamlit-1.40+-red.svg" alt="Streamlit">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

---

## 📌 项目简介

KnowFlow 是一款基于 **LangChain + LangGraph + Milvus** 的企业级 RAG 智能问答系统，支持：

- 🔍 **动态 RRF 混合检索**：向量检索 + BM25 关键词检索，权重根据查询特征自适应调整
- 🧠 **LLM 记忆压缩**：多轮对话自动压缩历史，支持 20+ 轮连续对话
- 🎯 **多级去重**：ID → 内容签名 → Token 重叠三级去重
- 📊 **Langfuse 全链路可观测**：追踪每一次检索、Rerank、LLM 调用
- 🔌 **LangGraph 编排**：RAG Pipeline + Agent ReAct 双 Graph 架构

## 🏗️ 架构设计

```
Streamlit UI → FastAPI → LangGraph Pipeline → Milvus + Redis + PostgreSQL
                         ├── Query Rewrite
                         ├── Hybrid Search (Dense + BM25)
                         ├── Dynamic RRF Fusion  ⭐ 核心创新
                         ├── Cross-Encoder Rerank
                         ├── Memory Consolidator
                         └── LLM Generation
```

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
# 编辑 .env，填入你的 LLM API Key

# 3. 安装依赖
cd backend && pip install -r requirements.txt
cd ../frontend && pip install -r requirements.txt
cd ..

# 4. 启动基础设施 (PostgreSQL + Redis + Milvus + MinIO)
docker compose up -d postgres redis milvus-standalone minio etcd

# 5. 启动后端 (新终端)
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 6. 启动前端 (新终端)
cd frontend
streamlit run app.py
```

访问:
- 前端 UI: http://localhost:8501
- 后端 API 文档: http://localhost:8000/docs
- MinIO 控制台: http://localhost:9001

### Docker 一键部署

```bash
docker compose up -d
```

## 📁 项目结构

```
KnowFlow/
├── backend/
│   ├── app/
│   │   ├── api/            # FastAPI 路由层
│   │   ├── core/           # 配置管理
│   │   ├── graph/          # LangGraph 编排
│   │   ├── models/         # SQLAlchemy 数据模型
│   │   ├── retrieval/      # 检索核心（动态 RRF / BM25 / Dense / Rerank）
│   │   ├── memory/         # 记忆压缩
│   │   ├── agent/          # Agent 工具
│   │   ├── services/       # 业务逻辑层
│   │   ├── utils/          # 工具函数
│   │   └── observability/  # Langfuse 集成
│   ├── migrations/         # Alembic 数据库迁移
│   ├── tasks/              # Celery 异步任务
│   ├── requirements.txt
│   └── main.py             # FastAPI 入口
├── frontend/
│   ├── pages/              # Streamlit 页面
│   ├── components/         # UI 组件
│   ├── utils/              # 工具函数
│   ├── requirements.txt
│   └── app.py              # Streamlit 入口
├── docker-compose.yml
├── .env.example
└── README.md
```

## 🔧 技术栈

| 层级 | 技术 |
|------|------|
| Web 框架 | FastAPI + Uvicorn |
| LLM 编排 | LangChain + LangGraph |
| 向量数据库 | Milvus |
| 关系数据库 | PostgreSQL |
| 缓存/队列 | Redis + Celery |
| 对象存储 | MinIO |
| 前端 | Streamlit |
| 可观测性 | Langfuse |

## ⭐ 核心特性详解

### 动态 RRF 融合

不同于业界固定权重（Vector:Keyword = 0.7:0.3），KnowFlow 基于四因子自适应：

1. **查询类型**：精确查询（代码/ID）→ 提升 Keyword；概念查询 → 提升 Vector
2. **查询长度**：短查询语义信号弱 → 提升 Keyword
3. **结果分布**：某路检索结果极少 → 自动补偿另一路
4. **分数方差**：方差大 = 区分度高 → 加权

### LLM 记忆压缩

当 Token 超过上下文窗口 50% 时，自动用低温度 LLM 对历史进行语义压缩。
压缩失败时退避为纯文本截断，保证服务可用性。

### 多级去重

Chunk ID → 内容 SHA256 签名 → Token 重叠系数（≥85% 时合并）

## 📄 License

MIT License

---

<p align="center">
  Made with ❤️ by KnowFlow Team
</p>
