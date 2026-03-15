# RAG 业务需求文档 v2.0

**版本**：v2.0  
**更新日期**：2026-03-14  
**目标读者**：后端开发、数据与策略  
**参考**：[rag.md](./rag.md)（初版，保留不动）

---

## 目录

1. [整体架构与分工](#1-整体架构与分工)
2. [数据与策略端规范（离线）](#2-数据与策略端规范离线)
3. [后端规范（在线）](#3-后端规范在线)
4. [SSE 协议扩展](#4-sse-协议扩展)
5. [关键待确认事项](#5-关键待确认事项)

---

## 1. 整体架构与分工

### 1.1 两条链路

```
【离线链路 · 数据与策略端】
原始文献 → 数据清洗(md) → chunks切分 → 向量化 → BM25索引 → 写入 PostgreSQL

【在线链路 · 后端】
用户query → 意图识别 → 混合检索 → RRF融合排序 → RAG Context组装 → LLM流式生成 → SSE推送
```

### 1.2 职责边界


| 模块                    | 负责方   | 说明                               |
| --------------------- | ----- | -------------------------------- |
| 原始文献获取与管理             | 数据与策略 | PDF/EPUB等原始资料                    |
| 数据清洗 → md             | 数据与策略 | 按2.2规范输出                         |
| chunks切分 + metadata组装 | 数据与策略 | 按2.3规范，产出LangChain Document对象    |
| 图片提取与描述               | 数据与策略 | 按2.4规范                           |
| embedding向量化          | 数据与策略 | 调用embedding模型，写入chunks.embedding |
| BM25索引构建              | 数据与策略 | 写入chunks.ts_vector（方案A）          |
| 写库                    | 数据与策略 | 按3.1表结构写入PostgreSQL              |
| 在线检索逻辑                | 后端    | 意图识别、混合检索、RRF                    |
| RAG Context组装         | 后端    | 按3.4规范                           |
| LLM调用与SSE推送           | 后端    | 含references事件                    |


---

## 2. 数据与策略端规范（离线）

### 2.1 知识库书目清单

以下 `source_id` 需提前约定固定 UUID，保证离线写库与在线检索一致。


| source_id（待定UUID）  | name            | content                           |
| ------------------ | --------------- | --------------------------------- |
| `src-001`（替换为UUID） | 生僻字库            | `rare_char`                       |
| `src-002`（替换为UUID） | 潘谷西、何建中《营造法式》解读 | `explanation`                     |
| `src-003`（替换为UUID） | 梁思成注释《营造法式》     | `original、annotation、explanation` |
| `src-004`（替换为UUID） | 王贵祥译注《营造法式》     | `original、annotation、explanation` |


> **Action**：数据端在正式建库前生成并固化上述UUID，同步给后端写入环境变量或配置文件。

### 2.2 数据清洗输出规范（md格式）

清洗后的md文件需满足以下要求：

- **标题层级**：保留原书层级，统一映射为 `# 卷` / `## 章` / `### 节`
- **正文**：删除页码、水印、页眉页脚、装饰性符号
- **图片**：将图片提取为独立文件（如 `{source_id}_{page}_{idx}.png`），在md正文中以 `![img:{image_id}]()` 占位，`image_id` 即后续 `images` 表的主键
- **编码**：UTF-8，行尾 LF
- **文件命名**：`{source_id}.md`

### 2.3 Chunk 切分与 metadata 规范

#### 切分策略


| 内容类型                                   | 切分方式            | 单chunk大小    | 重叠        |
| -------------------------------------- | --------------- | ----------- | --------- |
| 正文（original / annotation/explanation/） | 语义段落优先，递归字符切分兜底 | ≤ 800 token | 100 token |
| 术语（explanation）                        | 每条术语一个chunk     | 不限          | 无         |
| 生僻字（rare_char）                         | 每个字条目一个chunk    | 不限          | 无         |


规则：

- 不在句子中间截断
- 单段落超过800 token时，按句切分后合并至上限
- 切分结果转为 `langchain_core.documents.Document` 对象，`page_content` 为文本，`metadata` 见下方

#### metadata 字段约定

**所有字段必须存在，无值时用空字符串或空列表填充，不可省略字段。**

```json
{
  "source_id": "uuid-字符串",
  "source_name": "梁思成注释《营造法式》",
  "content_type": "text",
  "chapter": "卷三",
  "section": "壕寨制度",
  "page_start": 45,
  "page_end": 46,
  "chunk_index": 12,
  "has_images": false,
  "image_ids": []
}
```


| 字段               | 类型           | 说明                         |
| ---------------- | ------------ | -------------------------- |
| `source_id`      | string       | 书目UUID，见2.1                |
| `source_name`    | string       | 书目全名                       |
| `content_type`   | string       | 枚举：原文、注释、译文、其他，用于后续元数据过滤   |
| `chapter`        | string       | 所属卷/章，无则填 `""`             |
| `section`        | string       | 所属节，无则填 `""`               |
| `chunk_index`    | int          | 在该书目内的全局序号，从0开始            |
| `has_images`     | bool         | 该chunk关联图片时为 `true`        |
| `has_annotation` | bool         | 该文本是否有注释                   |
| `image_ids`      | list[string] | 关联的 images 表主键列表，无图时为 `[]` |
| `annotation`     | list[string] | 关联的 images 表主键列表，无图时为 `[]` |
| `image_ids`      | list[string] | 关联的 images 表主键列表，无图时为 `[]` |
| `image_ids`      | list[string] | 关联的 images 表主键列表，无图时为 `[]` |


### 2.4 图片处理规范

- **提取**：从原始文献中提取所有插图、表格截图
- **存储**：优先使用对象存储（如S3/MinIO），将URL写入 `images.url`；无对象存储时写入 `images.data`（BYTEA）
- **描述**：数据端为每张图片生成文字描述（`description` 字段），供无多模态能力时的文本备用检索
- **命名**：图片文件名格式 `{source_id}_{page}_{idx}.png`，`image_id` 为写库时生成的UUID

### 2.5 向量化规范

- 使用兼容 OpenAI client 的 embedding 模型（具体模型名称和维度由数据端确认后同步给后端）
- 对 `Document.page_content` 进行向量化，写入 `chunks.embedding` 列
- 向量维度需写入对应 `documents.vector_dim` 字段，供后端建索引参考

### 2.6 BM25 索引规范

**采用方案A（PostgreSQL tsvector + GIN索引）**：

- 数据端在写入chunks时，同步将 `content` 经中文分词后生成 `tsvector`，写入 `chunks.ts_vector`
- 中文分词推荐使用 `jieba`，分词后拼接空格再转 `tsvector`（PostgreSQL 原生tsvector对中文支持有限）
- 写入示例（Python伪代码）：

```python
import jieba
words = " ".join(jieba.cut(chunk_content))
# 在psycopg2中执行：
# UPDATE chunks SET ts_vector = to_tsvector('simple', %s) WHERE id = %s
```

> 若数据端评估后认为方案A效果不足，可改用方案B（单独bm25_index表存储BM25序列化索引），需提前与后端对齐查询接口。

---

## 3. 后端规范（在线）

### 3.1 新增数据库表 DDL

在现有 `users / sessions / messages` 表基础上，新增以下三张表。

> **前置条件**：需先在PostgreSQL中启用 `pgvector` 扩展：`CREATE EXTENSION IF NOT EXISTS vector;`

**documents 表**（书目信息，由数据端写入，后端只读）：

```sql
CREATE TABLE documents (
    id          VARCHAR(50) PRIMARY KEY,   
    name        TEXT        NOT NULL,
    metadata    JSONB                       -- 存储与业务逻辑相关的，比如包含的内容类型，用于后续元数据过滤
    description TEXT,
    vector_dim  INT,                        -- embedding维度，数据端写入，后端建索引时使用
    created_at  BIGINT      NOT NULL
);
```

**chunks 表**（核心RAG数据，由数据端写入，后端只读）：

```sql
-- 向量维度读取document，实际由数据端确认后替换
CREATE TABLE chunks (
    id          VARCHAR(50) PRIMARY KEY,
    document_id VARCHAR(50) NOT NULL REFERENCES documents(id), -- 来
    content     TEXT        NOT NULL,
    metadata    JSONB       NOT NULL,       -- 见2.3 metadata规范
    embedding   vector(1536),              -- pgvector，维度取documents表维度列
    ts_vector   TSVECTOR,                  -- BM25用，jieba分词后写入
 
    chunk_index INT         NOT NULL,
    created_at  BIGINT      NOT NULL
);

CREATE INDEX idx_chunks_embedding  ON chunks USING ivfflat (embedding vector_l2_ops) WITH (lists = 100);
CREATE INDEX idx_chunks_tsvector   ON chunks USING GIN (ts_vector);
CREATE INDEX idx_chunks_document   ON chunks (document_id);
CREATE INDEX idx_chunks_source     ON chunks USING GIN (metadata jsonb_path_ops);  -- 支持按source_id过滤
```

**images 表**（图片数据，由数据端写入，后端按需读取）：

```sql
CREATE TABLE images (
    id          VARCHAR(50) PRIMARY KEY,
    document_id VARCHAR(50) NOT NULL REFERENCES documents(id),
    page        INT,
    name        TEXT，                      --图名
    description TEXT,                      -- 图片内容文字描述，供文本备用检索
    url         TEXT,                      -- 对象存储URL（优先）
    data        BYTEA,                     -- 无URL时存二进制
    created_at  BIGINT      NOT NULL
);

CREATE INDEX idx_images_document ON images (document_id);
```

### 3.2 在线检索流程总览

```
用户 query
    │
    ▼
[Step 1] 意图识别（非流式LLM调用，约200-500ms）
    │  返回 query_type、source_ids[]、need_images、rewritten_query
    │
    ▼
[Step 2] 并行混合检索（按source_ids过滤 + 并行执行）
    ├── 向量检索：embedding相似度（L2距离），候选top 20
    └── BM25检索：tsvector全文匹配，候选top 20
    │
    ▼
[Step 3] RRF 融合排序 → 最终 top_k（默认5）
    │
    ▼
[Step 4] 按需加载关联图片
    │  need_images=true 时，查 images 表取对应记录（url 或 data）
    │
    ▼
[Step 5] 组装 RAG Context + 构建 references 列表
    │
    ▼
[Step 6] LLM 流式生成（stream=True）
    │
    ▼
SSE推送：meta → message(×N) → references → done
```

### 3.3 意图识别规范

**调用方式**：非流式，`stream=False`，超时建议设为 5s。

**System Prompt 要求**：

- 告知LLM当前可用的知识库列表（书目名称 + source_id）
- 要求输出合法JSON，不得有多余文字

**LLM 输出格式（JSON Schema）**：

```json
{
  "query_type": "term | content | image_related | mixed",
  "source_ids": ["uuid1", "uuid2"],
  "need_images": false,
  "rewritten_query": "改写后的查询，用于向量检索（可选，无改写时与原query相同）"
}
```


| 字段                | 类型           | 说明                                                            |
| ----------------- | ------------ | ------------------------------------------------------------- |
| `query_type`      | string       | `term`=术语查询，`content`=内容问答，`image_related`=涉及图样/图示，`mixed`=混合 |
| `source_ids`      | list[string] | 预计需要检索的书目UUID列表；空列表表示检索全部                                     |
| `need_images`     | bool         | 是否需要加载图片                                                      |
| `rewritten_query` | string       | 用于向量检索的改写query，改善语义匹配效果                                       |


**降级策略**：意图识别失败（超时/解析JSON失败）时，`source_ids=[]`（检索全库），`need_images=false`，`rewritten_query=原query`，不阻断主流程。

### 3.4 混合检索实现

#### 向量检索（L2距离）

```sql
SELECT
    id, content, metadata, chunk_index,
    embedding <-> %s::vector AS distance
FROM chunks
WHERE
    (%s::text[] IS NULL OR metadata->>'source_id' = ANY(%s::text[]))
ORDER BY distance
LIMIT 20;
```

参数：`(query_embedding, source_ids, source_ids)`

#### BM25检索（tsvector全文）

```sql
SELECT
    id, content, metadata, chunk_index,
    ts_rank(ts_vector, query) AS rank
FROM chunks,
     to_tsquery('simple', %s) AS query
WHERE
    ts_vector @@ query
    AND (%s::text[] IS NULL OR metadata->>'source_id' = ANY(%s::text[]))
ORDER BY rank DESC
LIMIT 20;
```

BM25查询词需做同样的jieba分词处理后拼接为 `tsquery` 格式（词之间用 `|` 或 `&` 连接）。

#### RRF 融合排序

```python
def rrf_score(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank)

# 合并向量检索和BM25检索结果，按chunk_id去重，累加RRF得分
def merge_rrf(vector_results, bm25_results, top_k=5):
    scores = {}
    for rank, chunk in enumerate(vector_results):
        scores.setdefault(chunk["id"], {"chunk": chunk, "score": 0})
        scores[chunk["id"]]["score"] += rrf_score(rank)
    for rank, chunk in enumerate(bm25_results):
        scores.setdefault(chunk["id"], {"chunk": chunk, "score": 0})
        scores[chunk["id"]]["score"] += rrf_score(rank)
    sorted_chunks = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
    return [item["chunk"] for item in sorted_chunks[:top_k]]
```

`top_k` 默认值为 **5**，通过环境变量 `RAG_TOP_K` 配置，策略端可调整。

### 3.5 RAG Context 组装规范

#### 文本 Context 格式

每个chunk拼接为以下格式，多个chunk之间用 `\n\n---\n\n` 分隔：

```
[1] 梁思成注释《营造法式》 · 卷三 · 壕寨制度（第45-46页）
{chunk.content}

---

[2] 王贵祥译注《营造法式》 · 卷三 · 壕寨制度（第102页）
{chunk.content}
```

#### 带图片 chunk 处理

- `need_images=true` 且 chunk 的 `has_images=true` 时，从 `images` 表加载对应图片
- 图片以多模态方式附加到LLM请求（`content` 数组中插入 `image_url` 类型元素）
- 若LLM不支持多模态，降级为在context中附加图片的 `description` 文字描述

#### System Prompt 模板

```
你是一个专业的中国古建筑知识问答助手，专注于《营造法式》相关内容。
请严格根据以下参考资料回答用户的问题。回答中引用具体来源时，使用 [数字] 标注，如 [1]、[2]。
若参考资料不足以回答，请明确说明。

参考资料：
{context}
```

#### references 列表构建

在组装context的同时，同步构建引用列表（与context中的 `[index]` 一一对应）：

```python
references = [
    {
        "index": i + 1,
        "chunk_id": chunk["id"],
        "source_name": chunk["metadata"]["source_name"],
        "chapter": chunk["metadata"]["chapter"],
        "section": chunk["metadata"]["section"],
        "page_start": chunk["metadata"]["page_start"],
        "page_end": chunk["metadata"]["page_end"],
    }
    for i, chunk in enumerate(top_chunks)
]
```

---

## 4. SSE 协议扩展

### 4.1 完整事件序列

在现有 `meta → message(×N) → done` 基础上，在 `done` 前新增 `references` 事件：

```
meta → message(×N) → references → title（仅新会话） → done
```

### 4.2 各事件格式

**meta 事件**（已有，无变化）：

```
event: meta
data: {"session_id": "uuid", "message_id": "uuid"}
```

**message 事件**（已有，无变化）：

```
event: message
data: {"content": "流式文本片段"}
```

**references 事件**（新增）：

```
event: references
data: {
  "references": [
    {
      "index": 1,
      "chunk_id": "uuid",
      "source_name": "梁思成注释《营造法式》",
      "chapter": "卷三",
      "section": "壕寨制度",
      "page_start": 45,
      "page_end": 46
    },
    {
      "index": 2,
      "chunk_id": "uuid",
      "source_name": "王贵祥译注《营造法式》",
      "chapter": "卷三",
      "section": "壕寨制度",
      "page_start": 102,
      "page_end": 103
    }
  ]
}
```

**title 事件**（已有，仅新会话时推送）：

```
event: title
data: {"title": "会话标题"}
```

**done 事件**（已有，无变化）：

```
event: done
data: {"status": "finished"}
```

### 4.3 无检索结果时的降级

若混合检索结果为空（未命中任何chunks），则：

- 不推送 `references` 事件
- System Prompt 去除参考资料部分，LLM基于自身知识回答
- `done` 事件中可附加标识：`{"status": "finished", "rag_used": false}`

### 4.4 messages 表 content 字段说明

assistant 消息的 `content` 字段存储纯文本回答（不含引用数据）。引用信息仅通过SSE实时推送，**不持久化到数据库**（前端按需缓存）。

---

## 5. 关键待确认事项

以下问题需数据与策略端、后端在开发前对齐，完成后更新本文档。


| 编号  | 问题                                                 | 负责方        | 状态  |
| --- | -------------------------------------------------- | ---------- | --- |
| Q1  | embedding 模型名称和向量维度是多少？（影响 `chunks.embedding` 列定义） | 数据端确认      | 待定  |
| Q2  | BM25方案确认：采用方案A（tsvector）还是方案B（自定义索引）？              | 数据端评估      | 待定  |
| Q3  | 图片存储方式：使用对象存储URL还是数据库BYTEA？                        | 数据端 + 后端运维 | 待定  |
| Q4  | 各 `source_id` 的UUID值？（需提前固化，写入后不可更改）               | 数据端生成，同步后端 | 待定  |
| Q5  | `top_k` 初始值及调整策略？（建议默认5，策略端实测后调整）                  | 策略端决策      | 建议5 |
| Q6  | 意图识别使用的LLM模型是否与主对话模型相同？                            | 后端 + 策略端   | 待定  |
| Q7  | LLM是否支持多模态（图片输入）？影响图片处理路径                          | 后端确认       | 待定  |


---

*文档维护：每次关键决策落地后，在对应行更新「状态」列。*