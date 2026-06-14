# 检索器优化

## 总体思路

将基于规则的意图检测升级为 LLM 驱动的意图识别，并引入可配置的多路检索与元数据过滤（`book_id`、`content_type` 等）。全量检索能力为 **六路**；按 LLM 返回的 `enabled_retrievers` 或默认配置启用子集。

## 〇、数据约定（实现前对齐）

### 〇.1 `documents.id`（真实主键，供过滤与意图映射）

`text_chunks.book_id` / `image_chunks.book_id` 外键指向 `documents.id`。当前库内书籍示例如下（以实际数据为准，实现时可配置或拉表缓存）：


| documents.id              | 说明（name 摘要）   |
| ------------------------- | ------------- |
| `terms_brief`             | 《法式》术语简要      |
| `yzfs_liang`              | 梁思成注释《营造法式》   |
| `yzfs_interpretation_rev` | 《营造法式》解读（修订版） |
| `yzfs_wang`               | 王贵祥译注《营造法式》   |
| `rare_chars`              | 《法式》生僻字库      |


**意图到书库的映射**：由 LLM 根据用户 **query** 判断应限定的 `book_ids`（可为空表示不限定、或多选）。不要求用户显式给出 id；模型结合书名/作者/主题与上表语义输出 `filters.book_ids`（值必须为已存在的 `documents.id`），校验失败时回退为「不限定 book」或规则默认。

### 〇.2 `content_type`（仅 `text_chunks`）

与 `app/models.py` 中 `**ContentTypeEnum`** 字符串一致（存库即用英文枚举值）：


| 枚举值                  | 含义   |
| -------------------- | ---- |
| `original_text`      | 原文   |
| `annotation`         | 注释   |
| `modern_translation` | 译文   |
| `interpretation`     | 解读   |
| `others_text`        | 其他文本 |


LLM 的 `filters.content_types` 必须使用上表中的值（可多选）。`image_chunks` **无** `content_type` 字段，图片侧过滤以 `**book_id`** 为主。

### 〇.3 `relation_type`（`relations` 表）

与 `**RelationTypeEnum**` 一致：`illustrates`、`annotates`。PRD 与实现均不另造别名。

---

## 一、LLM 意图识别模块设计

将意图定义为枚举，并附上所需过滤参数与启用的检索路：


| 意图类型                       | 触发示例        | 可用过滤条件 (filters)                                         | 推荐启用的检索路                                 | 备注                      |
| -------------------------- | ----------- | -------------------------------------------------------- | ---------------------------------------- | ----------------------- |
| `rare_char`                | 生僻字读音、部件拆解等 | `book_ids`: 常含 `rare_chars`                              | `text_vec`, `text_kw`（及按需 `text_toc_kw`） | 生僻字库对应 `rare_chars`     |
| `term_explain`             | 术语释义、含义、读音等 | 可无；或 `book_ids` 含 `terms_brief`                          | `text_toc_kw`, `text_vec`, `text_kw`     | 术语简要书可强关联 `terms_brief` |
| `original_and_translation` | 原文、译文、出处句   | `content_types`: 如 `original_text`, `modern_translation` | `text_kw` 为主，可配 `text_vec`               | 与 ContentTypeEnum 对齐    |
| `image_by_text`            | 找图、示意图      | 一般不限 book；可 `book_ids`                                   | `img_toc_kw`, `img_content_kw`           | **不查文本表**               |
| `specific_book`            | 指定译本/注释本章节  | `book_ids`: 如 `yzfs_liang`, `yzfs_wang`                  | 文本三路按需全开                                 | 由 query 推断译本            |
| `complex`                  | 多跳、对比、图文混合  | 子查询各自 filters                                            | 子查询合并前按子意图启用各路                           | 见 1.3                   |


### 1.2 LLM 调用方式

- **模型**：轻量模型，输出严格 JSON；模型名等由 **环境变量** 配置。
- **输出结构**（字段名固定，便于 Pydantic 校验）：

```json
{
  "query": "用户原始问题",
  "intents": [
    {
      "type": "original_and_translation",
      "filters": {
        "book_ids": ["yzfs_liang"],
        "content_types": ["original_text", "modern_translation"]
      },
      "enabled_retrievers": ["text_toc_kw", "text_vec", "text_kw"]
    }
  ],
  "is_complex": false,
  "sub_queries": []
}
```

- `**sub_queries**`：当 `is_complex === true` 时填写；每项含 `query` 及嵌套的 `type` / `filters` / `enabled_retrievers`（结构同单意图）。
- **校验与回退**：Pydantic 校验；非法 `book_ids` / 未知枚举则剔除非法项；若整份输出无效则回退到 **默认意图**（见下节）。

### 1.3 默认配置（无 LLM 或校验失败）

- **默认启用检索路**：五路并行检索全开（与 二、2.2 一致），`relation` 是否参与见 三、3.2。
- **默认不过滤**：`book_ids`、`content_types` 为空表示不限定。

### 1.4 复合问题拆分与合并

- **拆分**：由 LLM 按并列、因果、顺序等切分；每个子查询独立意图与 filters。
- **合并**：各子查询检索结果 **去重**（`type:id`）后做 **加权 RRF**；首版子查询权重均为 `1.0`，后续可按强调词等迭代。

---

## 二、多路检索器抽象与过滤器（核心工作）

### 2.1 统一接口

```python
class BaseRetriever:
    def retrieve(self, query: str, filters: dict, k: int) -> list[dict]:
        """
        filters 示例：
          {"book_ids": ["yzfs_liang", "rare_chars"], "content_types": ["original_text"]}
        仅 text_chunks 使用 content_types；image 仅 book_ids（及实现时约定的其他键）。
        返回与现有 hybrid_search 行结构兼容的 dict 列表。
        """
```

### 2.2 六路检索器定义与 id


| 内部 id            | 类名（示意）                       | 数据源与逻辑                                                                                         | 并行          |
| ---------------- | ---------------------------- | ---------------------------------------------------------------------------------------------- | ----------- |
| `text_toc_kw`    | TextTitleKeywordRetriever    | `text_chunks.toc_tsvector` + tsquery；过滤 `book_ids`、`content_types`                             | 是           |
| `text_vec`       | TextContentVectorRetriever   | 现有 `_text_vector_search` 扩展 `book_ids`、`content_types`                                         | 是           |
| `text_kw`        | TextContentKeywordRetriever  | 现有 `_text_keyword_search` 对应 SQL 扩展 `book_ids`、`content_types`（`ts_vector`）                    | 是           |
| `img_toc_kw`     | ImageTitleKeywordRetriever   | `image_chunks.toc_tsvector` + tsquery；过滤 `book_ids`                                            | 是           |
| `img_content_kw` | ImageContentKeywordRetriever | `image_chunks.ts_vector` + tsquery；过滤 `book_ids`                                               | 是           |
| `relation`       | RelationRetriever            | 依主检索得到的 chunk / 或查询解析出的实体，从 `relations` 拉关联块；可过滤 `relation_type` ∈ {`illustrates`,`annotates`} | **否**，见 3.2 |


**全量共六路**；运行时由 `enabled_retrievers` 或默认配置 **子集启用**。除 `relation` 外 **五路之间并行执行**（同一意图、同一子查询内）。

### 2.3 与旧代码关系

- 现有 `app/rag/retriever.py` 中文本/图关键词走 `ts_vector`；本方案将 **标题/目录** 与 **正文** 关键词拆为 `text_toc_kw` / `text_kw` 与 `img_toc_kw` / `img_content_kw`，需在 SQL 层区分列并实现统一 `filters`。

---

## 三、混合检索入口重构

### 3.1 函数签名

```python
def hybrid_search_v2(
    query: str,
    intent_result: dict | None = None,
    k_per_retriever: int = 5,
    k_final: int = 10,
    with_relations: bool = False,
) -> dict:
    """
    intent_result:
      {
        "intents": [{"type": str, "filters": dict, "enabled_retrievers": list[str]}],
        "sub_queries": [...]   # 可选，complex 时使用
      }
    intent_result is None 时：默认启用五路并行（relation 见下），filters 为空。
    """
```

### 3.2 执行流程

1. **解析意图**：无传入或非法则使用 1.3 默认配置。
2. **复合问题**：若存在 `sub_queries`，对每个子查询递归调用 `hybrid_search_v2`（不再拆 `sub_queries`），得到多组 items，再 **去重 + RRF** 合并为 `k_final`。
3. **单意图 / 单子查询**：
  - 根据 `enabled_retrievers` 取 **五路中启用子集**，**并行**调用 `retrieve`（共享同一 `filters` 与 `query`；必要时嵌入向量可复用一次 `embed_query`）。
  - 各路结果 **加权 RRF**（权重表与现 `lane_weights` 思路一致，扩展为五路 + 预留）。
  - `**relation`**：**不与其他五路同一批次并行**。推荐策略二选一（实现时选一种写死或配置）：
    - **A**：仅当 `with_relations=True` 或意图显式启用 `relation` 时，在主候选 `chunk_id` 集合上调用 RelationRetriever / 或复用 `_enrich_with_relations` 并尊重 `book_ids` 限制关联范围；
    - **B**：RelationRetriever 仅作为「扩展路」，在已有 top-k 主结果之后补充关联块（与现 enrich 对齐）。
4. **返回**：与现网结构兼容，并增加 `source_retriever`、`intent_type` 等便于调试字段（可选 `debug_info`）。

### 3.3 返回示例（字段扩展）

```json
{
  "items": [
    {
      "id": "uuid",
      "type": "text|image",
      "content": "...",
      "metadata": {},
      "score": 0.89,
      "source_retriever": "text_kw",
      "intent_type": "original_and_translation"
    }
  ],
  "relations": [],
  "debug_info": {
    "intent": {},
    "retriever_scores": {}
  }
}
```

---

## 四、性能与成本

### 4.1 减少 LLM 调用

规则前置：如含「原文」且可映射到 `content_types` / 固定书库时，可直接构造 filters 跳过 LLM（与现 `detect_query_intent` 可并存）。

### 4.2 数据库

- 为过滤字段建立/确认索引：`text_chunks(book_id, content_type)`，`image_chunks(book_id)`。
- 可按需对部分 `content_type` 建部分索引（如 `original_text`）。

### 4.3 复合问题

子查询是否 **复用主查询 embedding** 由实现定；首版可对子查询共享同一向量以控延迟与费用，并在 `debug_info` 中标记。

---

## 五、实现检查清单

- Pydantic：`IntentPayload`、`filters`、`enabled_retrievers` 枚举含六路 id。
- `book_ids` 仅允许 `documents` 中存在的 id（或配置白名单）。
- `content_types` 仅允许 `ContentTypeEnum` 值。
- `relation_type` 仅允许 `RelationTypeEnum` 值。
- 五路并行、`relation` 阶段与 `with_relations` 行为与 3.2 一致。
- 离线评测脚本与 `hybrid_search` 调用入口同步升级。

