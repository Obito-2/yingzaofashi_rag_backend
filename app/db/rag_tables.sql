-- RAG 业务表 DDL 脚本
-- 执行前确保已连接目标数据库
-- psql postgresql://postgres:lchgjt88@test-db-postgresql.ns-q5nnz4bx.svc:5432/postgres

-- 1. 启用 pgvector 扩展（支持 vector 类型和向量索引）
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. content_type 枚举（chunk 正文类型）
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'content_type_enum') THEN
        CREATE TYPE content_type_enum AS ENUM ('原文', '注解', '译文', '其他');
    END IF;
END$$;

-- 3. documents 表（原始知识库书籍信息，数据端写入，后端只读）
CREATE TABLE IF NOT EXISTS documents (
    id                UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    name              TEXT        NOT NULL,
    authors           TEXT[],
    publish_info      TEXT,                          -- 出版社、出版时间等版权信息
    metadata          JSONB,                         -- 备用业务元数据
    chunks_count      INTEGER     NOT NULL DEFAULT 0,
    vector_dimensions INTEGER     NOT NULL DEFAULT 1536,
    created_at        BIGINT      NOT NULL DEFAULT EXTRACT(EPOCH FROM NOW())::BIGINT * 1000,
    updated_at        BIGINT      NOT NULL DEFAULT EXTRACT(EPOCH FROM NOW())::BIGINT * 1000
);

-- 4. chunks 表（文本块，预计 <10k 条，数据端写入，后端只读）
CREATE TABLE IF NOT EXISTS chunks (
    id             UUID               PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id    UUID               NOT NULL REFERENCES documents(id),
    content        TEXT               NOT NULL,
    metadata       JSONB,                             -- 原始 metadata 备份
    embedding      vector(1536),                      -- 语义向量，维度与 documents.vector_dimensions 一致
    ts_vector      TSVECTOR,                          -- jieba 分词后的关键字索引
    content_type   content_type_enum,
    toc_path       TEXT[],                            -- 目录层级路径，可空
    has_images     BOOLEAN            NOT NULL DEFAULT FALSE,
    has_annotation BOOLEAN            NOT NULL DEFAULT FALSE,
    annotation     TEXT[],                            -- 关联注释内容，取自 metadata
    created_at     BIGINT             NOT NULL DEFAULT EXTRACT(EPOCH FROM NOW())::BIGINT * 1000,
    updated_at     BIGINT             NOT NULL DEFAULT EXTRACT(EPOCH FROM NOW())::BIGINT * 1000
);

-- chunks 向量检索索引（HNSW，余弦相似度，无需预训练即可插入查询）
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks USING hnsw(embedding vector_cosine_ops);

-- chunks 关键字检索索引（GIN，加速 tsvector @@ tsquery 查询）
CREATE INDEX IF NOT EXISTS idx_chunks_ts_vector ON chunks USING GIN(ts_vector);

-- chunks document_id 外键索引
CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);

-- 5. images 表（图片信息，预计 <1k 条，数据端写入，后端只读）
CREATE TABLE IF NOT EXISTS images (
    id           UUID    PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id  UUID    REFERENCES documents(id),   -- 可空，关联书籍
    name         TEXT    NOT NULL,
    description  TEXT,
    url          TEXT,
    binary_data  BYTEA,                              -- 可空，图片二进制数据
    created_at   BIGINT  NOT NULL DEFAULT EXTRACT(EPOCH FROM NOW())::BIGINT * 1000,
    updated_at   BIGINT  NOT NULL DEFAULT EXTRACT(EPOCH FROM NOW())::BIGINT * 1000
);

-- 6. chunk_images 关联表（chunk 与 image 的多对多关系）
CREATE TABLE IF NOT EXISTS chunk_images (
    chunk_id  UUID  NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    image_id  UUID  NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    PRIMARY KEY (chunk_id, image_id)
);

-- chunk_images 反向查询索引
CREATE INDEX IF NOT EXISTS idx_chunk_images_image_id ON chunk_images(image_id);
