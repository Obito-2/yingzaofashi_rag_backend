"""
RAG 知识库表迁移脚本
- 删除旧表：chunk_images, chunks, images, documents
- 创建新表：documents, text_chunks, image_chunks, relations
- 创建索引：B-tree / GIN / HNSW
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.connect import execute_query

DROP_SQLS = [
    "DROP TABLE IF EXISTS chunk_images CASCADE;",
    "DROP TABLE IF EXISTS chunks CASCADE;",
    "DROP TABLE IF EXISTS images CASCADE;",
    "DROP TABLE IF EXISTS documents CASCADE;",
]

CREATE_SQLS = [
    # 确保 pgvector 扩展已启用
    "CREATE EXTENSION IF NOT EXISTS vector;",

    # documents
    """
    CREATE TABLE documents (
        id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        name            TEXT NOT NULL,
        authors         TEXT[],
        other_metadata  JSONB,
        content         TEXT,
        created_at      BIGINT NOT NULL DEFAULT (EXTRACT(EPOCH FROM now())::BIGINT * 1000),
        updated_at      BIGINT NOT NULL DEFAULT (EXTRACT(EPOCH FROM now())::BIGINT * 1000)
    );
    """,

    # text_chunks
    """
    CREATE TABLE text_chunks (
        chunk_id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        content_type     VARCHAR(30),
        chunk_size       INTEGER,
        main_text        TEXT NOT NULL,
        book_id          UUID NOT NULL REFERENCES documents(id),
        closest_title    VARCHAR(500),
        toc_path         TEXT[],
        search_text      TEXT,
        ts_vector        TSVECTOR,
        other_metadata   JSONB,
        embedding_values vector(1024),
        created_at       BIGINT NOT NULL DEFAULT (EXTRACT(EPOCH FROM now())::BIGINT * 1000),
        updated_at       BIGINT NOT NULL DEFAULT (EXTRACT(EPOCH FROM now())::BIGINT * 1000)
    );
    """,

    # image_chunks
    """
    CREATE TABLE image_chunks (
        image_id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        title            VARCHAR(500),
        image_uri        TEXT,
        local_path       TEXT,
        alt_text         TEXT,
        caption          TEXT,
        book_id          UUID REFERENCES documents(id),
        closest_title    VARCHAR(500),
        toc_path         TEXT[],
        search_text      TEXT,
        ts_vector        TSVECTOR,
        embedding_values vector(1024),
        format           VARCHAR(20),
        created_at       BIGINT NOT NULL DEFAULT (EXTRACT(EPOCH FROM now())::BIGINT * 1000),
        updated_at       BIGINT NOT NULL DEFAULT (EXTRACT(EPOCH FROM now())::BIGINT * 1000)
    );
    """,

    # relations
    """
    CREATE TABLE relations (
        relation_id   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        source_type   VARCHAR(30) NOT NULL,
        source_id     UUID NOT NULL,
        target_type   VARCHAR(30) NOT NULL,
        target_id     UUID NOT NULL,
        relation_type VARCHAR(30) NOT NULL,
        created_at    BIGINT NOT NULL DEFAULT (EXTRACT(EPOCH FROM now())::BIGINT * 1000)
    );
    """,
]

INDEX_SQLS = [
    # text_chunks 索引
    "CREATE INDEX idx_text_chunks_book_id ON text_chunks USING btree (book_id);",
    "CREATE INDEX idx_text_chunks_content_type ON text_chunks USING btree (content_type);",
    "CREATE INDEX idx_text_chunks_ts_vector ON text_chunks USING gin (ts_vector);",
    "CREATE INDEX idx_text_chunks_embedding ON text_chunks USING hnsw (embedding_values vector_cosine_ops);",

    # image_chunks 索引
    "CREATE INDEX idx_image_chunks_book_id ON image_chunks USING btree (book_id);",
    "CREATE INDEX idx_image_chunks_ts_vector ON image_chunks USING gin (ts_vector);",
    "CREATE INDEX idx_image_chunks_embedding ON image_chunks USING hnsw (embedding_values vector_cosine_ops);",

    # relations 索引
    "CREATE INDEX idx_relations_source ON relations USING btree (source_id);",
    "CREATE INDEX idx_relations_target ON relations USING btree (target_id);",
]


def run_migration():
    print("=" * 50)
    print("开始 RAG 表迁移")
    print("=" * 50)

    print("\n[1/3] 删除旧表 ...")
    for sql in DROP_SQLS:
        table_name = sql.split("EXISTS")[1].split("CASCADE")[0].strip()
        execute_query(sql)
        print(f"  已删除: {table_name}")

    print("\n[2/3] 创建新表 ...")
    for sql in CREATE_SQLS:
        first_line = sql.strip().split("\n")[0].strip()
        execute_query(sql)
        print(f"  已执行: {first_line[:60]}")

    print("\n[3/3] 创建索引 ...")
    for sql in INDEX_SQLS:
        idx_name = sql.split("INDEX")[1].split("ON")[0].strip()
        execute_query(sql)
        print(f"  已创建索引: {idx_name}")

    print("\n" + "=" * 50)
    print("迁移完成！")
    print("=" * 50)


if __name__ == "__main__":
    run_migration()
