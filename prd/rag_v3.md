
**业务整体架构**

```
【离线链路 · 数据与策略端】
原始文献 → 数据清洗(md) → chunks切分（metadata丰富，包括章节信息注释和图片等） → 向量化 → BM25索引 → 写入 PostgreSQL数据库

【在线链路 · 后端】
用户query → 意图识别 → 混合检索 → RRF融合排序 → RAG Context组装 → LLM流式生成 → SSE推送
```

**RAG业务后端表设计**


1. 原始知识库documents表（书籍信息，目前3条，数据端写入，后端读）
    id uuid
    name text notbull
    authores text[]
    版权相关如出版社、出版时间等 text
    metadata JOSONB 备份业务相关元数据
    chunks count 包含的chunks数量
    向量维度
    创建时间
    更新时间

2. chunks表 （chunks信息，预计小于10k条，数据端写入，后端读）
    id
    document_id 外键，关联书籍表
    正文 text notnull
    metadata JSONB 原始metadata备份
    向量 维度取自关联的documents维度列，用于语义检索
    ts_vector TSVECTOR 用于关键字检索
    content_type：（取自metadata，固定枚举类型：原文、注解、译文、其他）
    toc_path text[]  可空
    has_images bool 不为空
    images_id text[] 关联的图片id，用于发送给llm组装回答，取自metadata
    has_annotation bool 不为空
    annotation text[] 关联的注释，取自metadata
    创建时间
    更新时间

3. 图片表 （images信息，预计小于1k条，数据端写入，后端读）
    id
    关联书籍id 外键 关联documents表 可空
    关联chunk_id 外键 关联chunks表 可空
    图名 text 不可空
    图片描述 可空
    url 可空
    二进制数据 可空
    创建时间
    更新时间


**离线数据处理**

1. 向量化，使用openai client自定义嵌入模型，向量化后存入chunks表embeeding列，查询使用相同的embeeding模型，计算寓意相似度。
2. chunks 表中，tsvector 类型的列来存储分词后的词位，并为其建立 GIN 索引。数据端在 Python 使用 jieba 库+自定义词表功能，对文档进行分词，造一个符合 tsvector 语法的字符串，再存入数据库。查询时：对用户输入的关键词也用 jieba +自定义词表分词，然后查询。