# 营造法式 RAG 问答后端

基于中国传统木建筑知识体系（《营造法式》及其译注/解读）的 RAG 智能问答系统后端，支持 Web 端、微信小程序及 AR 设备多端接入。

## 技术栈

| 层级 | 技术 |
|------|------|
| 语言 | Python 3.11 |
| Web 框架 | FastAPI 0.135.1 |
| 数据库 | PostgreSQL 16 + pgvector（向量检索） |
| AI Agent | LangGraph（检索→摘要→决策循环） |
| LLM | OpenAI 兼容接口（SEU / DashScope） |
| 包管理 | uv |

## 项目结构

```
yingzaofashi_rag_backend/
├── app/                          # 核心应用代码
│   ├── main.py                   # FastAPI 入口，路由注册
│   ├── connect.py                # 数据库连接管理
│   ├── models.py                 # 数据模型（ORM）
│   ├── agent/                    # LangGraph Agent
│   │   ├── graph.py              # 图构建（retrieve → summarize → decide）
│   │   ├── nodes.py              # 各节点实现
│   │   ├── state.py              # 状态定义
│   │   ├── prompts.py            # LLM 提示词模板
│   │   └── langgraph_app.py      # LangGraph CLI 加载入口
│   ├── api/                      # API 路由
│   │   ├── auth.py               # 用户认证
│   │   ├── sessions.py           # 会话管理
│   │   ├── chat.py               # 聊天服务（SSE 流式）
│   │   ├── messages.py           # 用户反馈
│   │   └── ar.py                 # AR 设备集成
│   ├── rag_v2/                   # 新版检索器（主力）
│   │   ├── hybrid_search.py      # 检索入口
│   │   ├── retrievers.py         # 五路检索器实现
│   │   ├── fusion.py             # 加权 RRF 融合
│   │   ├── relations.py          # 图文关联扩展
│   │   ├── intent_llm.py         # LLM 意图识别
│   │   └── schemas.py            # 数据结构定义
│   └── rag/                      # 旧版检索器（离线评测对比用）
├── data/                         # 评测数据集
├── docs/                         # 文档
│   ├── AR_INTEGRATION.md         # AR 设备集成指南
│   ├── CITATION_EXAMPLE.md       # 引文追踪示例
│   └── prd/                      # PRD 需求文档
├── experiments/                  # 实验数据与报告
├── resources/                    # 静态资源（结巴分词用户词典）
├── scripts/                      # 工具脚本
│   ├── agent_smoke_test.py       # Agent 冒烟测试
│   ├── build_retriever_experiment_dataset.py  # 评测数据集构建
│   ├── retriever_offline_eval.py  # 检索器离线评测
│   ├── run_agent_langsmith.py    # Agent 本地调试（LangSmith trace）
│   ├── test_rag_v2_langsmith.py  # rag_v2 本地调试（LangSmith trace）
│   └── langgraph_dev.ps1         # LangGraph 开发服务器启动脚本
├── pyproject.toml                # 项目配置与依赖
├── langgraph.json                # LangGraph CLI 配置
└── .env                          # 环境变量（不入库）
```

## 系统架构

```
用户请求 → FastAPI → LangGraph Agent
                       ├── Gate: 意图过滤（闲聊直接回复）
                       ├── Retrieve: LLM 意图识别 + 五路并行检索 + RRF 融合
                       ├── Summarize: 检索结果摘要压缩
                       ├── Decide: 信息充分性判断（不足则继续检索）
                       └── Final Answer: 流式 SSE 输出
```

### 检索策略

- **五路主检索**：目录关键词（text_toc_kw）、文本向量（text_vec）、正文关键词（text_kw）、图像目录关键词（img_toc_kw）、图像内容关键词（img_content_kw）
- **加权 RRF 融合**：各路检索结果按配置权重进行倒数排名融合
- **LLM 意图识别**：根据用户问题自动判定意图类型、过滤条件（book_ids / content_types）和启用检索器
- **图文关联扩展**：主检索结果后按 relations 表扩展关联图像/文本块

## 快速开始

### 环境要求

- Python 3.11 ~ 3.12
- PostgreSQL 16（需安装 pgvector 扩展）
- [uv](https://docs.astral.sh/uv/) 包管理工具

### 1. 安装依赖

```bash
uv sync
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env`（如不存在则手动创建），填入以下配置：

```env
# 数据库连接
DB_URL=postgresql://user:password@localhost:5432/yingzaofashi

# LLM API（二选一）
SEU_API_KEY=your_api_key
SEU_BASE_URL=https://your-api-endpoint/v1
# 或
DASHSCOPE_API_KEY=your_api_key
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# 模型配置
CHAT_MODEL_NAME=qwen-plus
INTENT_LLM_MODEL=qwen-turbo          # 意图识别轻量模型（可选，默认 qwen-turbo）

# Agent 配置
AGENT_GATE_MODE=auto                 # Gate 模式：auto / on / off
AGENT_MAX_RETRIEVE_DEPTH=3           # 最大检索深度
AGENT_CLUES_CHAR_THRESHOLD=20000     # 线索字符触发摘要阈值

# LangSmith（可选，用于 trace 调试）
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=yingzaofashi
```

### 3. 启动服务

```bash
# 方式一：直接启动 FastAPI
uv run uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload

# 方式二：启动 LangGraph Agent Server（支持 Chat UI 调试）
.\scripts\langgraph_dev.ps1 --port 8123
```

## API 接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/v1/auth/*` | POST | 用户注册/登录 |
| `/api/v1/sessions/*` | GET/POST/DELETE | 会话管理 |
| `/api/v1/chat/` | POST | 聊天（SSE 流式响应） |
| `/api/v1/chat/regenerate` | POST | 重新生成回答 |
| `/api/v1/messages/*` | POST | 用户反馈（点赞/踩） |
| `/ar/chat` | POST | AR 设备专用接口（无需认证，SSE 流式） |

## 工具脚本

| 脚本 | 用途 |
|------|------|
| `agent_smoke_test.py` | Agent 路由逻辑与图编译验证 |
| `retriever_offline_eval.py` | 检索器离线评测（HR@K / MRR / NDCG），支持 `--backend v1/v2` |
| `build_retriever_experiment_dataset.py` | 基于数据库内容构建评测数据集 |
| `run_agent_langsmith.py` | 本地运行 Agent 并在 LangSmith 查看 trace |
| `test_rag_v2_langsmith.py` | 本地运行 rag_v2 检索并在 LangSmith 查看 trace |

## 开发说明

- **包管理**：使用 `uv` 管理依赖，`uv.lock` 锁定版本
- **数据库**：主库为 PostgreSQL，`text_chunks` 表使用 `pgvector` 的 `ts_vector` 字段存储文本向量
- **检索器版本**：`rag_v2` 为当前主力，`rag` 保留用于离线评测对比
- **LangGraph**：支持 `langgraph dev` 启动本地 Agent Server，可通过 Web UI 交互调试

## 许可证

内部项目，仅供学习与研究使用。